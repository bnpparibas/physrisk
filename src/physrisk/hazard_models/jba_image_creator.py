import asyncio
from dataclasses import dataclass
import io
import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import aiohttp
import numpy as np
import PIL.Image as Image
from lxml import etree

from physrisk.api.v1.hazard_image import TileNotAvailableError
from physrisk.kernel.hazards import Hazard, HazardKind
from physrisk.kernel.hazard_model import HazardImageCreator, Tile

from physrisk.hazard_models.api_event_loop import get_loop, run
from physrisk.hazard_models.credentials_provider import (
    CredentialsProvider,
    EnvCredentialsProvider,
)

logger = logging.getLogger(__name__)


@dataclass
class TileSet:
    name: str
    release_date: str
    resolution: str
    projection: str


TileSpec = Tuple[int, int, int]  # (z, x, y)


class JBAImageCreator(HazardImageCreator):
    """Create images by calling out to JBA WMTS."""

    def __init__(
        self,
        credentials: Optional[CredentialsProvider] = None,
    ):
        self.credentials = (
            credentials if credentials is not None else EnvCredentialsProvider()
        )
        templates_tiles, templates_legends = self._get_urls_from_capability()
        self.templates_tiles: dict[str, str] = templates_tiles
        self.templates_legends: dict[str, str] = templates_legends
        self.tileset = TileSet("WR30", "202512", "30m", "4326")

    def create_image(
        self,
        resource_id: str,
        scenario: str,
        year: int,
        format="PNG",
        colormap: str = "heating",
        tile: Optional[Tile] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        index_value: Optional[Union[str, float]] = None,
    ):
        def expand(tile: Tile, size=512):
            f = size // 256
            return [
                Tile(tile.z + (f.bit_length() - 1), tile.x * f + dx, tile.y * f + dy)
                for dy in range(f)
                for dx in range(f)
            ]

        try:
            loop = get_loop()
            if index_value is None:
                index_value = 1500
            tiles = run(
                self._fetch_all_tiles(resource_id, int(index_value), expand(tile)),
                loop=loop,
            )
            stitched = self._stitch_tiles(tiles)
        except Exception as e:
            # if we are creating a whole image that does not exist, we log the error
            # and return a empty image; but if creating a tile we let the error propagate
            # because many map controls expect an HTTPException in such cases.
            if tile is None:
                logger.exception(e)
                stitched = Image.fromarray(np.array([[0]]), mode="RGBA")
            else:
                if isinstance(e, KeyError):
                    raise TileNotAvailableError(e.args[0]) from e
                else:
                    raise

        # we can optionally turn the image back into depths with the legend
        # and then we can apply a custom colour map
        # map_defn = colormap_provider.colormap(colormap)

        # def get_colors(index: int):
        #     return map_defn[str(index)]

        # rgba = self._to_rgba(data, get_colors, min_value=min_value, max_value=max_value)
        # image = Image.fromarray(rgba, mode="RGBA")

        image_bytes = io.BytesIO()
        stitched.save(image_bytes, format=format)
        return image_bytes.getvalue()

    def get_info(
        self, resource_id: str, scenario: str, year: int
    ) -> Tuple[Sequence[Any], Sequence[Any], str, str, Optional[int]]:
        index_values = [20, 50, 100, 200, 500, 1500]
        return (index_values, index_values, "return period", "years", 12)

    def _default_index_display_name(
        self, hazard_class: Type[Hazard], indicator_id: str
    ):
        if hazard_class.kind == HazardKind.ACUTE:
            return "return period"
        else:
            return "threshold"

    def _default_index_units(self, hazard_class: Type[Hazard], indicator_id: str):
        if hazard_class.kind == HazardKind.ACUTE:
            return "years"
        if indicator_id in [
            "days_wbgt_above",
            "mean_degree_days/above/index",
            "weeks_water_temp_above",
        ]:
            return "°C"
        else:
            return ""

    def _to_rgba(  # noqa: C901
        self,
        data: np.ndarray,
        get_colors: Callable[[int], List[int]],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        nodata_lower: Optional[float] = None,
        nodata_upper: Optional[float] = None,
        nodata_bin_transparent: bool = False,
        min_bin_transparent: bool = False,
    ) -> np.ndarray:
        """Convert the data to an RGBA image using values provided by get_colors.
        We are particular about min and max values, ensuring that these get their own indices
        from the colormap. Thee rules are:
        0: value is nodata
        1: value <= min_value
        2: min_value < value < (max_value - min_value) / 253
        254: (max_value - min_value) / 253 <= value < max_value
        255 is >= max_value

        Args:
            data (np.ndarray): Two dimensional array.
            get_colors (Callable[[int], Tuple[int, int, int]]): When passed an integer index in range 0:256, returns RGB components as integers in range 0:256.
            min_value (Optional[float]): Minimum value. Defaults to None.
            max_value (Optional[float]): Maximum value. Defaults to None.
            nodata_lower (Optional[float], optional): If supplied, values smaller than or equal to nodata_lower threshold are considered nodata. Defaults to None.
            nodata_upper (Optional[float], optional): If supplied, values larger than or equal to nodata_upper threshold are considered nodata. Defaults to None.
            nodata_bin_transparent (bool, optional): If True make no_data bin transparent. Defaults to False.
            min_bin_transparent (bool, optional): If True make min_bin transparent. Defaults to False.

        Returns:
            np.ndarray: RGBA array.
        """  # noqa

        red: np.ndarray = np.zeros(256, dtype=np.uint32)
        green: np.ndarray = np.zeros(256, dtype=np.uint32)
        blue: np.ndarray = np.zeros(256, dtype=np.uint32)
        a: np.ndarray = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            (red[i], green[i], blue[i], a[i]) = get_colors(i)
        if nodata_bin_transparent:
            a[0] = 0
        if min_bin_transparent:
            a[1] = 0
        mask_nodata = np.isnan(data)
        if nodata_lower:
            mask_nodata = data <= nodata_lower
        if nodata_upper:
            mask_nodata = (
                (mask_nodata | (data >= nodata_upper))
                if mask_nodata is not None
                else (data >= nodata_upper)
            )

        if min_value is None:
            min_value = np.nanmin(data)
        if max_value is None:
            max_value = np.nanmax(data)

        mask_ge_max = data >= max_value
        mask_le_min = data <= min_value

        np.add(data, -min_value, out=data)
        np.multiply(data, 253.0 / (max_value - min_value), out=data)
        np.add(data, 2.0, out=data)  # np.clip seems a bit slow so we do not use

        result: np.ndarray = data.astype(np.uint8, casting="unsafe", copy=False)
        del data

        if mask_nodata is not None:
            result[mask_nodata] = 0
            del mask_nodata

        result[mask_ge_max] = 255
        result[mask_le_min] = 1
        del mask_ge_max, mask_le_min

        final = (
            red[result]
            + (green[result] << 8)
            + (blue[result] << 16)
            + (a[result] << 24)
        )
        return final

    def _get_urls_from_capability(self):
        # async is not necessary, but we follow the same pattern
        loop = get_loop()
        with aiohttp.TCPConnector(loop=loop) as conn:
            identifiers, template_tiles, template_legends = {}, {}, {}

            async def get_capability():
                try:
                    async with aiohttp.ClientSession(
                        connector=conn, connector_owner=False
                    ) as session:
                        set_name = "WR30_202512_30m_4326"
                        url = f"https://jbavision.jbarisk.com/cog/WMTS/{set_name}?service=WMS&request=GetCapabilities&version=1.3.0"
                        async with session.get(
                            url=url,
                            proxy=self.credentials.proxies()["https"],
                            auth=aiohttp.BasicAuth(
                                self.credentials.jba_vision_username(),
                                self.credentials.jba_vision_password(),
                            ),
                        ) as resp:
                            resp.raise_for_status()
                            e_tree = etree.fromstring(await resp.text())
                            ns = {
                                "wmts": "http://www.opengis.net/wmts/1.0",
                                "ows": "http://www.opengis.net/ows/1.1",
                                "xlink": "http://www.w3.org/1999/xlink",
                            }
                            layers = e_tree.xpath("//wmts:Layer", namespaces=ns)
                            for layer in layers:
                                template_tile = layer.xpath(
                                    "wmts:ResourceURL[@format='image/png']",
                                    namespaces=ns,
                                )[0].get("template")
                                template_legend = layer.xpath(
                                    "wmts:Style/wmts:LegendURL[@format='image/png']",
                                    namespaces=ns,
                                )[0].get("{" + ns["xlink"] + "}href")
                                title_text = layer.xpath("ows:Title", namespaces=ns)[
                                    0
                                ].text
                                identifier = layer.xpath(
                                    "ows:Identifier", namespaces=ns
                                )[0].text
                                identifiers[title_text] = identifier
                                template_tiles[title_text] = template_tile
                                template_legends[title_text] = template_legend
                except Exception as e:
                    logger.exception(e)

            run(get_capability(), loop=loop)
            return template_tiles, template_legends

    async def _fetch_tile(
        self, session: aiohttp.ClientSession, url: str
    ) -> Image.Image:
        """Download a single tile and return it as a Pillow Image."""
        async with session.get(url) as resp:
            resp.raise_for_status()  # raise on HTTP errors
            data = await resp.read()  # raw bytes
            return Image.open(io.BytesIO(data)).convert("RGBA")  # ensure RGBA

    async def _fetch_all_tiles(
        self, resource_id: str, return_period: int, tile_specs: List[TileSpec]
    ):
        """Download all tiles concurrently and return them in the same order."""
        try:
            async with aiohttp.ClientSession(
                proxy=self.credentials.proxies()["https"],
                auth=aiohttp.BasicAuth(
                    self.credentials.jba_vision_username(),
                    self.credentials.jba_vision_password(),
                ),
            ) as session:
                tasks = []
                for z, x, y in tile_specs:
                    url = self.templates_tiles[
                        self._identifier(self.tileset, resource_id, return_period)
                    ].format(TileMatrix=z, TileCol=x, TileRow=y)
                    tasks.append(self._fetch_tile(session, url))
                return await asyncio.gather(*tasks)
        except Exception as e:
            logger.exception(e)

    def _stitch_tiles(self, tiles, grid=(2, 2)):
        """
        Assemble a list of Pillow images into one image.

        Parameters
        ----------
        tiles : list[Image.Image]
            Tiles ordered row‑wise (left → right, top → bottom).
        grid : tuple[int, int]
            (cols, rows) of the final mosaic.

        Returns
        -------
        Image.Image
            The combined image.
        """
        cols, rows = grid
        if len(tiles) != cols * rows:
            raise ValueError("Number of tiles does not match the grid size")

        # Assume all tiles have the same dimensions
        tile_w, tile_h = tiles[0].size
        combined = Image.new("RGBA", (cols * tile_w, rows * tile_h))

        for idx, tile in enumerate(tiles):
            col = idx % cols
            row = idx // cols
            combined.paste(tile, (col * tile_w, row * tile_h))

        return combined

    def _identifier(self, tile_set: TileSet, resource_id: str, return_period: int):
        if resource_id == "jba_undefended_riverine":
            return f"{tile_set.name}_{tile_set.release_date}_FLRF_U_RP{return_period}_RD_{tile_set.resolution}_{tile_set.projection}"
        elif resource_id == "jba_undefended_pluvial":
            return f"{tile_set.name}_{tile_set.release_date}_FLSW_U_RP{return_period}_RD_{tile_set.resolution}_{tile_set.projection}"
        elif resource_id == "jba_sop":
            return f"{tile_set.name}_{tile_set.release_date}_DRAS_D_VE_{tile_set.resolution}_{tile_set.projection}"

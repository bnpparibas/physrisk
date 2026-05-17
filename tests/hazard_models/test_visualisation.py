from physrisk.kernel.hazard_model import Tile
import pytest
from esg_physrisk.hazard_models.jba_image_creator import JBAImageCreator


# https://jbavision.jbarisk.com/cog/tiles/9/265/176.png?LAYERS=853_WR30_202512_30m_4326:WR30_202512_FLRF_U_RP1500_RE_30m_4326

# Use the 30m Global Inland Flood Map by setting the country_code parameter to WR30, or
# Continue using individual country map layers by setting country_code to the relevant country.
# https://jbavision.jbarisk.com/cog/WMTS/WR30_202512_30m_4326


#@pytest.mark.skip("Requires live connection")
def test_JBA_image_creator(load_credentials):
    creator = JBAImageCreator()
    creator.create_image(
        "jba_undefended_riverine", "historical", -1, tile=Tile(0, 0, 0)
    )
    assert creator is not None
    # for country_code in ["WR30", "FR5C", "GB", "BE", "US"]:
    # Path("tile.png").write_bytes(resp.content)

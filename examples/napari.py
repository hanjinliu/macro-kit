import napari
from napari.layers import Image
from magicgui import magicgui
from magicgui.widgets import PushButton, Container
from scipy import ndimage as ndi
from macrokit import Macro, Symbol

# Create macro instance
macro = Macro()

# Register Image layer type to tell Symbol how to display it in macro.
Symbol.register_type(Image, lambda layer: f"viewer.layers['{layer.name}']")

# ... or if you are used to Expr, then this code will be the better solution.
#
# from macrokit import Expr
#
# def layer_expr(layer: Image):
#     expr = Expr("getitem", [Expr("getattr", [Symbol("viewer"),
#                                              Symbol("layers")]),
#                             layer.name])
#     return expr
#
# Symbol.register_type(Image, layer_expr)

viewer = napari.Viewer()

@magicgui
@macro.record
def gaussian_filter(img: Image, sigma: float = 1.0, output_name: str = ""):
    out = ndi.gaussian_filter(img.data, sigma)
    viewer.add_image(out, name=output_name)

# A push button to show macro
btn = PushButton(text="Create Macro")
btn.changed.connect(lambda: print(macro.optimize()))

widget = Container(widgets=[gaussian_filter, btn], labels=False)

viewer.window.add_dock_widget(widget)

napari.run()
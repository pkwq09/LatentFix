
def rgba(c: str):
    from matplotlib import colors as mcolors
    return mcolors.to_rgba(c)

def rgb(c: str):
    from matplotlib import colors as mcolors
    return mcolors.to_rgb(c)

color_map = {

             'source_motion': (223/255.0, 92/255.0, 100/255.0, 1.0),
             'source': (223/255.0, 92/255.0, 100/255.0, 1.0),
             'target_motion': (124/255.0, 213/255.0, 149/255.0, 1.0),
             'input': (124/255.0, 213/255.0, 149/255.0, 1.0),
             'target': (124/255.0, 213/255.0, 149/255.0, 1.0),
             'generation': (209/255.0, 162/255.0, 70/255.0, 1.0),
             'generated': (209/255.0, 162/255.0, 70/255.0, 1.0),
             'denoised': rgba('purple'),
             'noised': rgba('darkgrey'),
             }
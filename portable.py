"""
The code uses the gooey library to create a GUI for the script. It allows users to select image sources ("pexels" and/or "bing"), enter categories to download, specify the number of images to download per class, and choose an output location.

The main function is decorated with @Gooey to create the GUI interface. It defines the command-line arguments using GooeyParser and processes the selected options. The args object contains the parsed arguments.

The code checks if the output location directory exists and creates it if it doesn't. It then proceeds to download images from Pexels or Bing based on the selected sources. The download_pexel_images function is called if "pexels" is selected, and the bing_downloader function is called if "bing" is selected.

Finally, the main function is called if the script is run directly.
"""

import os
import sys
from gooey import Gooey, GooeyParser
from charset_normalizer import md__mypyc
from u2net_test import process_saliency
from u2net_portrait_test import process_potraits
from u2net_human_seg_test import process_hum_seg
@Gooey()
def main():
    parser = GooeyParser(description="U2NET Portable")
    parser.add_argument(
        "MODE",
        metavar="MODE",
        help="MODE",
        widget="Listbox",
        nargs="+",
        choices=["Background Removal","Human Segmentation"],
    )
    parser.add_argument("INPUT_LOCATION", help="Choose the input location of Images", widget="DirChooser",default = r"C:\Users\AKHIL\projects\U2Net\U-2-Net\test_data\test_images")
    parser.add_argument("OUTPUT_LOCATION", help="Choose the location to save images", widget="DirChooser",default = r"C:\Users\AKHIL\projects\U2Net\U-2-Net-Portable\temp\b")
    args = parser.parse_args()

    if not os.path.exists(args.OUTPUT_LOCATION):
        os.makedirs(args.OUTPUT_LOCATION)

    if "Background Removal" in args.MODE:
        process_saliency(args.INPUT_LOCATION,os.path.join(args.OUTPUT_LOCATION,"BG_REMOVAL"))

    elif "Potrait" in args.MODE:
        process_potraits(args.INPUT_LOCATION,os.path.join(args.OUTPUT_LOCATION,"POTRAIT"))

    elif "Human Segmentation" in args.MODE:
        process_hum_seg(args.INPUT_LOCATION,os.path.join(args.OUTPUT_LOCATION,"HUMAN_SEG"))

if __name__ == "__main__":
    main()
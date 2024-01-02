# ArtificialColour

http://artificialcolour.pythonanywhere.com/

This project introduces a colour-combining tool based on the GAN framework, aiming to enhance the creative workflow when selecting colours. This was spurred by the promising avenues opened up by advancements in technology and artificial intelligence. Despite the existence of a few machine learning models online, there is still a notable scarcity of thorough documented projects in the published literature that effectively address the challenges associated with colour combination solutions.

To train the generative model, a dataset was curated, comprising colour palettes derived from frames of animated films. The colours in the dataset were described using the perceptually uniform CIELUV colour system. The tool is accessible via a web page, promoting cross-platform usability.

User interactions with the web page generate a new dataset, which can be utilised for ongoing training of the generative model or applied in similar projects. An in-depth analysis of colour palettes in the frames dataset was conducted, focusing on hue, chroma, and luminance distribution. Additionally, the stability of the GAN was examined by studying its losses across multiple training epochs.

The algorithms introduced, along with the two datasets serve as a foundation for future experimental research in the realm of colour theory. The web tool not only enhances colour combination solutions but also contributes with a constantly expanding dataset for further exploration in related projects.

Notes on Watershed Delineation
==============================

# Requirements

There are two basic requirements for delineating a watershed:

  * A Digital Elevation Map (DEM)
  * The outlet location

There are some considerations to be aware of when selecting data for a delineation.

## Selecting a DEM

The delineation will be the most accurate if the elevation is in the same units as the geographic coordinates. Otherwise the difference in scale may lead to rounding errors when slope is computed, especially when the grid size is very small. Since elevation is typically in units of meters this means that the DEM should be in a projected coordinate system, also with units of meters. A transverse mercator projection from the appropriate UTM zone would be an appropriate choice. The datum is likely insignificant across the region being delineated, but if there is concern that there may be significant and unaccounted for geoid separation over the range of the study, a local datum could be selected as well.

Raw elevation data collected by remote sensing often contains trees and other structures, the height of which does not affect the hydrology. An extreme example would be a flat grassland with trees overhanging a river, such as the area north of Boulder and east of the mountains. If the trees are included, then the river will be the highest point in the landscape, wreaking havoc with the delineation.  Typically an elevation product will be filtered to show the best approximation of the topography underneath. However, it is important to make sure that this process has actually been completed with the dataset you use! There are a variety of terms that are used to designate datasets with different levels of processing. While these terms are often used interchangeably, the title does give some indication of the nature of the dataset:

  * Digital Surface Model (DSM): More or less raw data expected to contain houses, trees, etc.
  * Digital Elevation Model (DEM): Best guess of what the terrain looks like based on filtering out identifyable vegetation and structures
  * Digital Terrain Model (DTM): Modifications made to make the hydrology of the dataset align with other, more reliable data. For example, a ground survey of a stream network might be used to modify elevations with the intention of making sure that the DTM agrees with the survey.

Sometimes other terms are used as well, so when using an unknown datasource it is advisable to read the documentation thoroughly so you know what you are getting.

# Process


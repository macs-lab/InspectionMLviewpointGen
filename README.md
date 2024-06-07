# InspectionMLviewpointGen

Requires `python 3.10.0`

## Dependencies required to run Segmentation with automatic point cloud preprocessing Jupiter notebook

Install dependencies.
```
pip3 install open3d==0.17.0 scikit-learn pytransform3d numpy bayesian-optimization ipykernel
```

## To run the Segmentation with automatic point cloud preprocessing (SPCP)

Open spcp.ipynb in src folder and follow the instructions on the python notebook

## Working
- The python notebook takes in an STL model as input and generates segmented point cloud (each color representing a segment) satisfying the camera parameters such as the field of view and the depth of field
- The point cloud is segmented initially using region growth and then using K-means clustering as the methodology
- The clustering happens in three stages, that is first to derive the regions after point cloud processing, next to segment these regions into planar segments, and then to divide the planar segments to fit them within the field of view
- For the second stage exponential search is used and for the third stage Bayesian Optimization is used to find the optimal K values
    
The image below shows the overall process:

![image](https://github.com/macs-lab/InspectionMLviewpointGen/assets/114765006/f8ae318d-1f0e-486c-9934-e148815460d0)


For the FOV segmentation bayesian optimization the updation of cost function is shown below:

![gif_bayesian optimization](https://github.com/macs-lab/InspectionMLviewpointGen/assets/114765006/04b6b449-ce76-48e9-b17a-bdbc99275c1a)

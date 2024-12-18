# InspectionMLviewpointGen

Requires `python 3.10.0`

## Dependencies required to run Bayesian Segmentation Jupiter notebook

Install dependencies.
```
pip3 install open3d==0.17.0 scikit-learn pytransform3d numpy==1.26.4 bayesian-optimization==2.0.2 ipykernel
```

## To run the Bayesian Segmentation

Open bayesian_segmentation.ipynb in src folder and follow the instructions on the python notebook

## Working
- The python notebook takes in an STL model as input and generates segmented point cloud (each color representing a segment) satisfying the camera parameters such as the field of view and the depth of field
- The point cloud is segmented using K-means clustering as the methodology
- The clustering happens in two stages, that is first to derive the planar segments and then to divide the planar segments to fit it within the field of view
- For the first stage exponential search is used and for the second Bayesian Optimization is used to find the optimal K values
    
The image below shows the overall process:

![image](https://github.com/macs-lab/InspectionMLviewpointGen/assets/114765006/0262e47c-11a2-4557-ab83-d5e0f68b5397)

For the FOV segmentation bayesian optimization the updation of cost function is shown below:

![gif_bayesian optimization](https://github.com/macs-lab/InspectionMLviewpointGen/assets/114765006/04b6b449-ce76-48e9-b17a-bdbc99275c1a)

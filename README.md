# DIP Term Project

### Description

Term project from Digital Image Processing lecture. A big topic is stitching pictures.



### Stitching two pictures

If you don't have any prior information, you can use the matching point information from `knnMatch` SIFT function.

The following is how to get the feature point coordinates from the DMatch object, which is the return value of the knnMatch function.

```python
for match in matches:
    p1 = kp1[match.queryIdx].pt
    p2 = kp2[match.trainIdx].pt
```

The value `p1[0]` is returned as a real array of x coordinate. If you check this, you can get distance matching point from the left. This information help to get what is left side picture is that.

![image](https://user-images.githubusercontent.com/109254266/179134979-e557f351-8b19-413a-acb8-2da1408a530b.png)

![image](https://user-images.githubusercontent.com/109254266/179135088-84ad8fe9-0d0c-4518-9b16-a14d09b55fc1.png)



### Stitching four horizon pictures

Stitch like this order : 1-2, 3-4, 1-2-3-4

![image](https://user-images.githubusercontent.com/109254266/179135674-443b5105-f60f-4257-b4ec-0f32f43f97cc.png)

![image](https://user-images.githubusercontent.com/109254266/179135742-142a56b6-0db7-4611-bdae-fd498796b37e.png)



### Stitching four unsorted picture

sorted by sum of x-coordinate and stitch pictures.



### Stitching four picture

Previously, only the x-coordinate was used, but here, the y-coordinate is also used to distinguish the top, bottom, left, and right.

![image](https://user-images.githubusercontent.com/109254266/179136571-1a3c6e73-b369-44bb-bf95-02568e13b3bf.png)

![image](https://user-images.githubusercontent.com/109254266/179136600-f8490970-4ad3-4bd9-8e3e-bb10ee152699.png)

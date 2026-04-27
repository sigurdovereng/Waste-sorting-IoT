# Design Decisions & Trade-offs

## 1. Two Models Instead of One
We used two separate models: a COCO object detector and a custom waste classifier.

This made development faster and more practical. The detector first finds the object in the image, then the classifier predicts the waste category. Training one end-to-end detector for custom waste classes would require a much larger labeled dataset and more training time.


## 2. Detection Confidence Threshold = 0.45
Only detections above 45% confidence are accepted.

A lower threshold increased false detections, while a higher threshold missed valid objects. 0.45 gave the best balance during testing.

## 3. Classification Confidence Threshold = 0.65
Predictions below 65% confidence are rejected.

This prevents uncertain classifications from activating the wrong LED and improves reliability in real-world use.


## 4. Classification Margin Threshold = 0.15
The top prediction must be at least 15% higher than the second-best prediction.

This reduces mistakes when two classes have very similar probabilities.

## 5. Minimum Detection Area = 2000
Very small bounding boxes are ignored.

Small detections were often background objects or noise. Filtering them improved crop quality and final classification accuracy.


## 6. On-device TensorFlow Lite Inference
The final classifier was converted to TensorFlow Lite and executed directly on the Raspberry Pi.

This reduced latency, removed cloud dependency, and allowed the system to run offline.


## 7. Custom Dataset Instead of Only Public Data
Public datasets were useful for initial training, but additional project-specific images were needed.

This helped reduce domain shift caused by different lighting conditions, camera angle, and background in the real setup.
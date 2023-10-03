from mlops import annotations, anomaly_detection, dataset, segmentation

class DefaultDebugger:
    def __init__(self, configs):
        self.configs = configs
    
    def run(self):
        print("check annotation...")
        annotation_result = annotations.check_annotation_format(self.configs)
        print("check dataset...")
        dataset_result = dataset.check_image_format(self.configs)
        print("check segmentation...")
        segmentation_result = segmentation.check_segmentation(self.configs)
        print("check anomaly_detection...")
        anomaly_detection_result = anomaly_detection.check_anomaly_detection(self.configs)
        return {
            "annotation":annotation_result,
            "dataset":dataset_result,
            "segmentation":segmentation_result,
            "anomaly_detection":anomaly_detection_result
        }
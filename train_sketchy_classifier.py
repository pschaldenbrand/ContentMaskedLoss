from sketchy.classifier import SketchyClassifier
print(SketchyClassifier.class_names)
print(SketchyClassifier.sketchy_img_dir)
classifier = SketchyClassifier()
classifier.train(500)
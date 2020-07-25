from sketchy.classifier import SketchyClassifier
print(SketchyClassifier.class_names)
print(SketchyClassifier.sketchy_img_dir)
classifier = SketchyClassifier(model='robustness')
classifier.train(200,batch_size=32)
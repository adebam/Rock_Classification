# Final Data Restructuring and Results
## Class Restructuring and Merging

The analysis in the previous section suggested that the primary limitation of the classification task may not lie in model architecture alone, but also in the structure and consistency of the dataset. Persistent confusion between specific class pairs indicated that some classes were not clearly separable visually, while the degradation in performance from training to validation and test suggested limited generalization.

Although class imbalance contributed to the problem, imbalance-aware strategies such as weighted focal loss with normal sampling and ordinary cross-entropy with weighted sampling improved performance without eliminating the recurring confusion between several well-represented classes. This suggested that class overlap and label definition were also important sources of error.

Based on the confusion-matrix analysis, class-wise F1 scores, and visual inspection of the images, the following classes were therefore merged:

| Original Classes | Restructured Class |
| :--- | :--- |
| **Class 2 + Class 13** | Merged class |
| **Class 18 + Class 19** | Merged class |
| **Class 21 + Class 22** | Merged class |

These merges reduce the original **22-class problem to 19 classes**.

An additional labeling inconsistency was identified for **Class 1** during the initial dataset exploration. Image filenames in the training split were associated with **Vug**, whereas images assigned to Class 1 in the validation and test splits were labeled **Arenaceous**. Visual inspection also showed a noticeable difference between the appearance of the Class 1 training images and those in the validation set. Because the same class identifier appeared to represent different petrographic categories across the dataset splits, Class 1 was removed rather than retained as a potentially inconsistent target.

After removing Class 1 and merging the three highly confused class pairs, the classification problem was reduced from **22 classes to 18 classes**.

The restructuring was applied consistently across the training, validation, and test directories while preserving the original dataset separately. The model was then retrained using the revised 18-class label structure so that performance could be evaluated under the simplified and more internally consistent classification scheme.

## Validation Performance Before and After Class Restructuring

<p align="center">
  <img src="../pictures/18Classes_vs_22classes_val.png" alt="Val_18vs22classes"><br>
  <i>Figure 1: Validation Results 18 classes vs 22 classes</i>
</p>

The figure compares the full validation trajectories of the original **22-class model (pink)** and the **restructured 18-class model (green)**. Showing the complete training history makes it possible to compare not only the peak scores, but also how each formulation behaved as training continued.

For the original **22-class** problem, validation performance improved rapidly during the first several epochs. The validation F1-score reached approximately **0.43** around epoch 5–6, after which performance became unstable and did not show sustained improvement. By epoch 10, the F1-score was lower than the value achieved around epoch 5, while validation loss had increased substantially from its minimum around epoch 4. This confirmed the earlier decision to retain the epoch-5 checkpoint as the best 22-class model rather than simply using the final epoch.

The **restructured 18-class model** followed a different pattern. Its validation F1 initially tracked the 22-class model closely, but continued improving after the point at which the 22-class model began to plateau. By the final epochs, the 18-class model reached approximately **0.47** validation F1, exceeding the best performance observed with the original class structure.

The improvement is even clearer in validation accuracy, which increased steadily to approximately **0.56** for the 18-class model, compared with roughly **0.46–0.47** for the 22-class model. Precision also continued to improve and reached approximately **0.52**, while recall remained consistently stronger throughout most of training.

The validation-loss curves provide additional evidence of the difference between the two formulations. For the 22-class model, loss reached an early minimum and then increased as training continued, suggesting that additional epochs were no longer improving generalization. In contrast, the 18-class model showed an overall downward trend in validation loss through the later epochs, indicating that useful learning was still occurring.

The full training trajectories show that the 22-class model reached its strongest validation performance early and subsequently degraded, which is why the epoch-5 checkpoint was selected. After restructuring the dataset to 18 classes, the model was able to continue improving beyond this point, ultimately achieving higher F1, accuracy, and precision together with a more favorable validation-loss trend.

These results support the hypothesis that some of the original class definitions were creating unnecessary ambiguity. Removing the inconsistent Class 1 and merging repeatedly confused classes appears to have produced a more learnable and internally consistent classification problem.

> **Note:** One caveat should still be stated: the 18-class and 22-class models solve different classification problems, so their numerical scores are not perfectly equivalent benchmarks. Some improvement is naturally expected when ambiguous classes are merged and the number of target categories is reduced. The important observation is therefore not only that the final score increased, but that the training behavior itself improved: the restructured model continued learning and generalizing after the original 22-class model had already plateaued and begun to degrade.

## Class-wise F1 Comparison: 18 Classes vs. 22 Classes

<p align="center">
  <img src="../pictures/18classesvs22classesf1score.png" alt="Val_18vs22f1score"><br>
  <i>Figure 2: F1-score by class, 18 classes vs 22 classes</i>
</p>
The figure compares the class-wise validation F1-scores of the original 22-class model (pink) with the restructured 18-class model (green). It provides a more detailed view of how the restructuring affected individual classes rather than considering only the overall validation metrics.

A direct class-to-class comparison should be made carefully because the two models do not solve exactly the same classification problem. Several original classes were merged—Class 2 with Class 13, Class 18 with Class 19, and Class 21 with Class 22—and Class 1 was removed because of the identified labeling inconsistency. Consequently, some class labels in the original 22-class model no longer have an independent equivalent in the 18-class model.

The class-wise F1 comparison also show that restructuring the dataset did not produce a uniform improvement across classes that remained unchanged. For many of these classes, F1-scores remained similar, while some improved slightly and others declined.

This suggests that the improvement in overall validation performance was not caused by a broad increase in the model's ability to recognize all petrographic classes. Instead, the primary benefit of restructuring appears to have come from removing an inconsistent class and eliminating several difficult decision boundaries by merging classes that were repeatedly confused.

Because the merged classes no longer exist as separate targets in the 18-class formulation, their scores cannot be compared directly with the corresponding 22-class results. The figure should therefore be interpreted together with the confusion matrices and overall validation metrics rather than as a direct class-by-class benchmark..

> **Note:** Because the class definitions changed, the most meaningful comparison is the overall behavior of the two models and the performance of classes that remained unchanged, rather than a strict one-to-one comparison across every class.

## Confusion Matrix Result on Validation Data

<p align="center">
  <img src="../pictures/18classes_confusionMaxtrixValidation.png" alt="Val_18ClassconfusionMatrix"><br>
  <i>Figure 3: F1-score by class, 18 classes vs 22 classes</i>
</p>

The confusion matrix for the restructured 18-class problem shows that merging the selected class pairs simplified some of the previously ambiguous decision boundaries, but did not eliminate the broader classification difficulty.

The merged **Class 2/Class 13** category shows relatively strong separation, with approximately 844 of 1,083 images classified correctly. This suggests that the combined category forms a reasonably coherent target after the distinction between the original two classes is removed.

The results for the other merged categories are less clear. The combined **Class 18/Class 19** category has approximately 482 correct predictions out of 1,056 images, while the merged **Class 21/Class 22** category has approximately 1,210 correct predictions out of 2,191 images. Both categories continue to be confused with several other classes, including with each other.

This indicates that merging removed the requirement for the model to distinguish between the original paired classes, but it did not fully resolve the underlying visual overlap in the dataset. The classification difficulty therefore extends beyond individual class pairs and may reflect broader similarities between petrographic categories, variation within classes, or inconsistencies in the dataset.

The confusion matrix also supports the earlier class-wise F1 analysis: restructuring did not suddenly improve the recognition of all remaining classes. Instead, its main effect appears to have been the removal of a small number of problematic decision boundaries and one known inconsistent class.

>Overall, restructuring simplified the label space but did not eliminate the fundamental class-separability problem. The remaining off-diagonal errors suggest that ambiguity exists not only within the merged class pairs but also between several broader petrographic categories.

## Final Test Result
The final 18-class model showed a substantial difference between training, validation, and test performance.

| Metric              | Training | Validation | Test    |
| :------------------ | :------- | :--------- | :------ |
| **Accuracy**        | 0.79546  | 0.55706    | 0.38550 |
| **Macro F1**        | 0.83806  | 0.46734    | 0.30072 |
| **Macro Precision** | 0.77005  | 0.52191    | 0.30694 |
| **Macro Recall**    | 0.94281  | 0.50699    | 0.34875 |

Performance decreases consistently from training → validation → test. Training accuracy reached approximately 79.5%, but fell to 55.7% on validation and 38.6% on the test set. The same pattern is even more apparent in macro F1, which decreased from 0.838 during training to 0.467 on validation and 0.301 on test.

The particularly large gap between training and test performance indicates that generalization remains a major limitation, even after restructuring the class definitions. The very high training macro recall of 0.943, compared with only 0.349 on the test set, suggests that the model learned the training examples effectively but was considerably less successful at identifying the same classes in unseen images.

The confusion matrix supports this interpretation. Although the merged classes remove several previously difficult within-pair distinctions, substantial off-diagonal confusion remains. In particular, the combined **Class 18/Class 19** and **Class 21/Class 22** categories are still confused with several other classes. This suggests that the classification difficulty extends beyond the specific pairs that were merged and may reflect broader visual overlap, within-class variability, or inconsistencies in the petrographic class definitions.

### Comparison with the Original 22-Class Model

The restructured model nevertheless produced better test metrics than the original 22-class formulation:

| Metric              | 22 Classes | 18 Classes | Change  |
| :------------------ | :--------- | :--------- | :------ |
| **Accuracy**        | 0.2817     | 0.3855     | +0.1038 |
| **Macro F1**        | 0.2346     | 0.3007     | +0.0661 |
| **Macro Precision** | 0.2578     | 0.3069     | +0.0491 |
| **Macro Recall**    | 0.2571     | 0.3488     | +0.0917 |

All four test metrics improved after restructuring, with accuracy increasing by approximately 10.4 percentage points and macro F1 by approximately 6.6 percentage points. This suggests that removing the inconsistent Class 1 and consolidating several highly confused class pairs made the classification problem more tractable.

However, these results should not be interpreted as a perfectly controlled model-to-model improvement, because the 18-class model solves a simpler and differently defined classification problem. Some improvement is naturally expected after reducing the number of target classes and removing difficult boundaries.

### Overall Interpretation
<p align="center">
  <img src="../pictures/18_classes_Final_test_results.png" alt="test_confusion_matrix"><br>
  <i>Figure 3: Confusion Matrix, Final Test Results</i>
</p>

The restructuring experiment therefore produced an important result: simplifying the label space improved overall performance, but did not resolve the underlying generalization problem.

### Color Distribution of Images in Classes across Train, Validation & Test Dataset
Image-property analysis revealed that the performance degradation from validation to test was accompanied by measurable within-class distribution shifts. For example, Class 14 training and validation images showed similar color and brightness distributions, whereas the test images exhibited very different RGB intensity and brightness and substantially saturation. This suggests that differences in image appearance between splits may have contributed to the reduced test-set generalization.
<p align="center">
  <img src="../pictures/split_comparison_class14_properties.png" alt="Class 14 distribution across split"><br>
  <i>Figure 4: Class 14 distribution across split</i>
</p>

<p align="center">
  <img src="../pictures/split_comparison_class2_class13_properties.png" alt="Class2_3 distribution across split"><br>
  <i>Figure 5: Class 14 distribution across split</i>
</p>

<p align="center">
  <img src="../pictures/split_distribution_shift_heatmap_22classes_spacious.png" alt="Heatmpa_all_classes"><br>
  <i>Figure 6: Class 14 distribution across split</i>
</p>


The heatmap compares how image properties differ between the training distribution and the validation/test splits for each class. Most validation cells are relatively close to zero, indicating that validation images are generally similar to the training data. In contrast, the test split shows larger shifts for several classes, particularly in color intensity, brightness, contrast, and saturation.
This suggests that part of the drop in test performance may be related to distribution shift: some test images have visual characteristics that differ from the images the model was trained on. However, the effect is not uniform across all classes, so distribution shift is likely one contributor rather than the only cause of poor generalization.



The original 22-class experiments showed that no single factor fully explained the poor performance. Imbalance-aware training improved results, confirming that class frequency mattered, while restructuring the labels produced a further gain, showing that ambiguous or overlapping class definitions also contributed to the problem. However, the persistent train–validation–test performance gap indicated that these changes did not fully resolve the generalization issue.
The split-distribution analysis provides additional evidence that the dataset itself is a major limitation. Validation images were generally closer to the training distribution, while several test classes showed larger shifts in color intensity, brightness, contrast, and saturation. This suggests that part of the test-performance degradation may be caused by distribution shift between the training and test images, in addition to class overlap and labeling ambiguity.

> **Note:** The final results suggest that the main limitation is not simply model architecture or class imbalance, but the overall consistency and separability of the dataset. Class restructuring improved the task, yet substantial class ambiguity and measurable differences in image properties across dataset splits remained, limiting generalization to unseen petrographic images.

Even if overall test performance is limited, can the model still identify a subset of predictions that are reliable?

## Confidence Analysis
The final evaluation showed that the classifier does not generalize equally well across all unseen images. The image-property analysis also suggested that some test samples differ from the training distribution, which may contribute to uncertain or unreliable predictions. Also the model predicted some images with high degree of probability, yet the image was classified incorrectly.

For deployment, this means that returning a class label for every image may be misleading. Instead, the model’s predicted probability can be used as a confidence measure, allowing low-confidence predictions to be flagged rather than automatically accepted, while also identifying high-confidence predictions that disagree with the assigned label as potential labeling inconsistencies for further review.

For each image, the predicted class is the class with the highest softmax probability, and the corresponding probability is treated as the model confidence. A prediction is accepted only when this confidence exceeds a selected threshold. That confidence score was then used as a filter: predictions above a selected cutoff were accepted, while lower-confidence predictions were flagged for review. The predicted classes themselves did not change; only the subset of predictions considered acceptable was adjusted.

### Getting a threshold of 75%

The analysis began with the 5,994 validation images. Without applying any confidence filter, the model matched the provided labels on 55.71% of the images. The minimum confidence threshold was then increased gradually from 50% to 90%, while tracking two quantities: the accuracy of the accepted predictions and the proportion of images that remained accepted.

A clear tradeoff emerged. As the confidence threshold increased, the accepted predictions became more accurate, but fewer images met the acceptance criterion.
At a 70% confidence threshold, approximately half of the images were accepted, with 69.07% accuracy among the accepted predictions. Increasing the threshold to 75% raised accepted accuracy to 70.95%, while retaining 45.71% of the images. At an 80% threshold, accepted accuracy increased further to 72.37%, but coverage fell to 39.37%.

<p align="center">
  <img src="../pictures/06_threshold_validation_tradeoff_summary.png" alt="ConfidencePlot"><br>
  <i>Figure 6: Confidence Plot for all classes</i>
</p>

A 75% confidence threshold was selected as the main operating point because it provided a practical balance between prediction reliability and coverage. This threshold was not derived from a formal optimization rule; rather, it was chosen as a reasonable tradeoff between accepting enough images and improving the accuracy of the accepted subset.

<p align="center">
  <img src="../pictures/06_threshold_75_per_class_validation.png" alt="ConfidencePlot"><br>
  <i>Figure 7: Confidence Plot per class</i>
</p>


The class-level results also showed that the confidence threshold did not behave uniformly across all classes. Some classes retained many high-confidence predictions with relatively strong accuracy, while others retained very few images or continued to show substantial disagreement with the provided labels. This indicated that the overall threshold-level accuracy could hide important differences in class-specific behavior.

### Confidence Results on the Test
The 75% confidence threshold, selected using the validation dataset, was then fixed and applied unchanged to the 2,786 test images using the same trained model and preprocessing pipeline. This allowed the threshold to be evaluated on previously unseen data without further adjustment.
| Result | Validation | Test |
| :--- | ---: | ---: |
| **Accuracy without rejection** | 55.71% | 38.55% |
| **Images accepted at 75% confidence** | 2,740 / 5,994 | 923 / 2,786 |
| **Coverage** | 45.71% | 33.13% |
| **Correct accepted predictions** | 1,944 | 440 |
| **Accepted predictions disagreeing with labels** | 796 | 483 |
| **Accuracy among accepted images** | 70.95% | 47.67% |

Applying the confidence threshold improved test accuracy among the accepted predictions from 38.55% to 47.67%, an increase of approximately 9.1 percentage points. This shows that model confidence contains useful information: higher-confidence predictions were more likely to agree with the provided labels than predictions considered as a whole.

However, the improvement was substantially smaller than on the validation set. At the same 75% threshold, validation accuracy among accepted images was 70.95%, compared with only 47.67% on the test set. Coverage also decreased from 45.71% on validation to 33.13% on test. Therefore, the same confidence threshold accepted fewer test images and those accepted predictions were considerably less reliable.

This result is consistent with the earlier evidence of distribution differences between the training/validation and test datasets. A confidence score of 75% therefore did not correspond to the same observed accuracy across the two evaluation sets. In other words, the model could still be highly confident on test images even when its prediction disagreed with the provided label.

The **high-confidence disagreements are particularly important. Of the 923 test images accepted at the 75% threshold, 483 (52.33%) disagreed with the provided labels**. These cases should not automatically be interpreted as model errors or labeling errors. Instead, they represent a useful subset for further investigation. Repeated high-confidence disagreements involving the same classes or visually similar images may indicate class overlap, distribution shift, or potential labeling inconsistencies.

> **Overall interpretation**: Confidence-based rejection improved the reliability of accepted predictions, but confidence alone was not sufficient to overcome the test-set generalization problem. The large difference between validation and test performance at the same threshold suggests that the model's confidence is not equally reliable across dataset splits. Nevertheless, confidence remains useful both for rejecting uncertain predictions and for identifying high-confidence disagreements that may warrant manual review.
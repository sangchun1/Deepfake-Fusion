# OpenFake baseline report summary

## 1) merged

- accuracy: 0.9552
- precision: 0.9586
- recall: 0.9514
- f1: 0.9550
- auc: 0.9906
- loss: 0.1761

## 2) by_generator

- num_experiments: 12
- mean_accuracy: 0.9696
- mean_precision: 0.9709
- mean_recall: 0.9682
- mean_f1: 0.9695
- mean_auc: 0.9949
- mean_loss: 0.0923
- best_accuracy: grok-2-image-1212 (0.9878)
- worst_accuracy: flux-1.1-pro (0.9350)
- best_auc: grok-2-image-1212 (0.9992)
- worst_auc: flux-1.1-pro (0.9834)

- best accuracy:
  - grok-2-image-1212: 0.9878
  - sdxl-epic-realism: 0.9859
  - gpt-image-1: 0.9816
  - midjourney-6: 0.9816
  - ideogram-3.0: 0.9790
- worst accuracy:
  - sd-3.5: 0.9678
  - hidream-i1-full: 0.9559
  - flux.1-dev: 0.9553
  - imagen-4.0: 0.9540
  - flux-1.1-pro: 0.9350

- best auc:
  - grok-2-image-1212: 0.9992
  - sdxl-epic-realism: 0.9989
  - midjourney-6: 0.9985
  - gpt-image-1: 0.9984
  - ideogram-3.0: 0.9976
- worst auc:
  - sd-3.5: 0.9943
  - flux.1-dev: 0.9926
  - hidream-i1-full: 0.9918
  - imagen-4.0: 0.9899
  - flux-1.1-pro: 0.9834

## 2) logo

- num_experiments: 12
- mean_accuracy: 0.8729
- mean_precision: 0.9447
- mean_recall: 0.7909
- mean_f1: 0.8543
- mean_auc: 0.9587
- mean_loss: 0.6336
- best_accuracy: gpt-image-1 (0.9511)
- worst_accuracy: flux-1.1-pro (0.7217)
- best_auc: gpt-image-1 (0.9892)
- worst_auc: flux-1.1-pro (0.8784)

- best accuracy:
  - gpt-image-1: 0.9511
  - flux.1-dev: 0.9475
  - ideogram-3.0: 0.9347
  - hidream-i1-full: 0.9307
  - imagen-4.0: 0.9285
- worst accuracy:
  - flux-mvc5000: 0.8552
  - grok-2-image-1212: 0.8227
  - midjourney-6: 0.8216
  - sd-3.5: 0.7896
  - flux-1.1-pro: 0.7217

- best auc:
  - gpt-image-1: 0.9892
  - flux.1-dev: 0.9882
  - ideogram-3.0: 0.9841
  - imagen-4.0: 0.9827
  - hidream-i1-full: 0.9822
- worst auc:
  - flux-mvc5000: 0.9516
  - midjourney-6: 0.9468
  - grok-2-image-1212: 0.9451
  - sd-3.5: 0.9217
  - flux-1.1-pro: 0.8784

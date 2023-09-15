import torch
import numpy as np

def accuracy(output, target, return_length=False):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    if return_length:
        return correct / len(target), len(target)
    else:
        return correct / len(target)
    
def class_avg_accuracy(output, target, return_length=False, num_classes=16):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        # print("pred, target:",pred, target)
        corrects = {}
        lengths = {}
        for c in range(num_classes):
            if not c in corrects:
                corrects[c] = 0
                lengths[c] = 0
            corrects[c] += torch.sum(pred[torch.where(target == c)] == c).item()
            lengths[c] += torch.sum(target == c).item()
    class_accuracy = [(corrects[c] / lengths[c]) if lengths[c] > 0 else 0.0 for c in corrects.keys()] ## setting it to 0.0 is error-prone
    avg_accuracy = sum(class_accuracy) / len(class_accuracy)
    # print("class_accuracy:", class_accuracy)
    if return_length:
        return avg_accuracy, class_accuracy, len(target)
    else:
        return avg_accuracy, class_accuracy

# def class_avg_accuracy_misclassifymat(output, target, num_classes=16):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         # print("pred, target:",pred, target)
#         corrects = {}
#         lengths = {}
#         misclassify_mat = torch.zeros((num_classes, num_classes), dtype=int)
#         for c in range(num_classes):
#             if not c in corrects:
#                 corrects[c] = 0
#                 lengths[c] = 0
#             corrects[c] += torch.sum(pred[torch.where(target == c)] == c).item()
#             lengths[c] += torch.sum(target == c).item()
#         for i in range(len(target)):
#             misclassify_mat[target[i], pred[i]] += 1
#     class_accuracy = [(corrects[c] / lengths[c]) if lengths[c] > 0 else 0.0 for c in corrects.keys()]
#     avg_accuracy = sum(class_accuracy) / len(class_accuracy)
#     # print("class_accuracy:", class_accuracy)
#     # print("misclassify_mat: ",misclassify_mat)

#     return avg_accuracy, class_accuracy, misclassify_mat

def class_avg_accuracy_misclassifymat(output, target, num_classes=16):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        # print("pred, target:",pred, target)
        corrects = np.zeros(num_classes, dtype=int)
        lengths = np.zeros(num_classes, dtype=int)
        misclassify_mat = torch.zeros((num_classes, num_classes), dtype=int)
        for c in range(num_classes):
            # if not c in corrects:
            #     corrects[c] = 0
            #     lengths[c] = 0
            corrects[c] += torch.sum(pred[torch.where(target == c)] == c).item()
            lengths[c] += torch.sum(target == c).item()
        for i in range(len(target)):
            misclassify_mat[target[i], pred[i]] += 1
    # class_accuracy = [(corrects[c] / lengths[c]) if lengths[c] > 0 else 0.0 for c in corrects.keys()]
    # avg_accuracy = sum(class_accuracy) / len(class_accuracy)
    # print("class_accuracy:", class_accuracy)
    # print("misclassify_mat: ",misclassify_mat)
    # return avg_accuracy, class_accuracy, misclassify_mat
    return list(corrects), list(lengths), misclassify_mat


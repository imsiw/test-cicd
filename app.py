import gradio as gr
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


def recognize(image):
    # Преобразуем картинку PIL в тензор
    input_tensor = preprocess(image)
    # Создаем вырожденную пачку из 1 элемента
    input_batch = input_tensor.unsqueeze(0)
    # Отключаем обучение
    with torch.no_grad():
        # Прогоняем данные через модель нейронной сети
        output_tensor = model(input_batch)
    # С помощью функции активации softmax получаем вероятности классов
    probabilities = torch.nn.functional.softmax(output_tensor[0], dim=0)
    # print(probabilities)
    # Выбираем 5 самых вероятных вариантов
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    # print(top5_catid)
    # print(top5_prob)
    result = []
    for i in range(top5_prob.size(0)):
        # Название класса
        label = labels[top5_catid[i]]
        # Вероятность класса
        probability = top5_prob[i].item()
        # print(label, probability)
        result.append(label + " " + str(probability))
    return "\n".join(result)


# Используем готовую модель, включаем режим инференса
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# Преобразование входных данных
preprocess = transforms.Compose([
    transforms.Resize(256),      # Масштабирование изображения к 256 точкам
    transforms.CenterCrop(224),  # Обрезка краев до размера 224x224 точки
    transforms.ToTensor(),       # Преобразование нормализованному тензору со значениями [0,1]
    transforms.Normalize(        # Стандартизация к статистике в датасете ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Названия классов изображений
# https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
with open("imagenet_classes.txt", "r") as f:
    labels = [s.strip() for s in f.readlines()]

# Создание приложения Gradio
demo = gr.Interface(
    fn=recognize,
    inputs=[gr.Image(type="pil")],
    outputs=["text"],
    api_name="predict"
)
demo.launch()

from PIL import Image
import numpy as np
import random
import math
import base64
import json

def getCircle(bg, fg=(255, 0, 0)):
    size = 15
    circle = np.reshape([
        bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg,
        bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg,
        bg, bg, bg, bg, bg, bg, fg, fg, fg, bg, bg, bg, bg, bg, bg,
        bg, bg, bg, bg, bg, fg, fg, fg, fg, fg, bg, bg, bg, bg, bg,
        bg, bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg,
        bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg,
        bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg, bg,
        bg, bg, bg, bg, bg, fg, fg, fg, fg, fg, bg, bg, bg, bg, bg,
        bg, bg, bg, bg, bg, bg, fg, fg, fg, bg, bg, bg, bg, bg, bg,
        bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg,
        bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg
    ], (size, size, 3))
    return size, circle

def getRectangle(bg, fg=(0, 255, 0)):
    size = 25
    circle = np.reshape([
        bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg,
        bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg,
        bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, bg, bg,
        bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg,
        bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg,
        bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg, bg,
    ], (size, size, 3))
    return size, circle

def getShape(label, x, y, size):
    return {
        "label": label,
        "line_color": None,
        "fill_color": None,
        "points": [[x, y], [x + size, y + size]],
        "shape_type": "rectangle"
    }

def createDummyImage(imageFileName):

    width = random.randint(300, 400)
    height = random.randint(200, 300)

    backgroundColor = (180, 230, 255)
    imageData = []
    for x in range(width):
        for y in range(height):
            imageData.append(backgroundColor)

    shapes = []
    imageMatrix = np.reshape(imageData, (height, width, 3))
    for i in range(3):
        size, circle = getCircle(backgroundColor)
        x = random.randint(10, width-10-size)
        y = random.randint(10, height-10-size)
        imageMatrix[y:y+size, x:x+size, :] = circle
        shape = getShape('Circle', x, y, size)
        shapes.append(shape)
    for i in range(3):
        size, rect = getRectangle(backgroundColor)
        x = random.randint(10, width-10-size)
        y = random.randint(10, height-10-size)
        imageMatrix[y:y+size, x:x+size, :] = rect
        shape = getShape('Rectangle', x, y, size)
        shapes.append(shape)

    imageData = []
    for y in range(height):
        for x in range(width):
            imageData.append(tuple(imageMatrix[y][x]))

    img = Image.new('RGB', (width, height))
    img.putdata(imageData)
    img.save(imageFileName, quality=95)

    version = "3.15.2"
    imageDataBase64 = ""

    with open(imageFileName, "rb") as im:
        imageDataBase64 = base64.b64encode(im.read()).decode('utf-8') # + '=='

    data = {
        "version": version,
        "flags": {},
        "shapes": shapes,
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": imageFileName,
        "imageData": imageDataBase64,
        "imageHeight": height,
        "imageWidth": width
    }

    jsonFileName = ''.join(imageFileName.split('.')[:-1]) + '.json'
    with open(jsonFileName, 'w+') as f:
        json.dump(data, f)


def createDummyData(n = 10, start = 0):

    for i in range(start, n):
        decimals = int(math.log10(n))
        fileName = '{}.jpg'.format(format(i, '0' + str(decimals)))
        createDummyImage(fileName)


random.seed(42)
createDummyData(100, 0)


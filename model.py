from main import *
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


lines = []
with open('../input/carnd/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples = []
train_measurements = []
train_marks = []
lines_count = len(lines)
train_lines_count = int(lines_count*0.8)
validation_lines_count = lines_count - train_lines_count
for line in lines[:train_lines_count]:
    # 0 center image, 1 left image, 2 right image, 3 measurement
    measurement = float(line[3])
    for i in range(3):
        current_path = line[i]
        #image = cv2.imread(current_path)
        #images.append(image)
        train_samples.append(current_path)
        factor = 0.0
        if i == 1:
            factor = 0.2
        if i == 2:
            factor = -0.2
        train_measurements.append(measurement + factor)
        train_marks.append(0)
# data augmentation
for i in range(train_lines_count):
    train_samples.append(train_samples[i])
    train_measurements.append(train_measurements[i])
    train_marks.append(1) # 1 for horizontal flipping


validation_samples = []
validation_measurements = []
validation_marks = []
for line in lines[train_lines_count:lines_count]:
    # 0 center image, 1 left image, 2 right image, 3 measurement
    measurement = float(line[3])
    current_path = line[0]
    validation_samples.append(current_path)
    validation_measurements.append(measurement)
    validation_marks.append(0)


train_generator = generator(train_samples, train_measurements, train_marks)
validation_generator = generator(validation_samples, validation_measurements, validation_marks)

model = car_net()
model.summary()

steps_per_epoch = len(train_samples) / 3
validation_steps = len(validation_samples) / 3

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
h = model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    nb_epoch=3)


print('Train Images:', len(train_samples))
print('Validation Images:', len(validation_samples))


plot_loss_and_accuracy(h.history)


SVG(model_to_dot(model).create(prog='dot', format='svg'))


HTML("""
<video width="320" height="160" controls>
  <source src="video.mp4">
</video>
""")


model.save('model.h5')


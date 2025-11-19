from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(rescale=1./255)

flow = data_gen.flow_from_directory(
    "data/train",
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical"
)

batch, labels = next(flow)

print("Batch shape:", batch.shape)
print("Labels:", labels[:5])
print("Classes:", flow.class_indices)

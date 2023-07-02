defmodule Mix.Tasks.TrainHandwrittenDigitsModel do
  @moduledoc """
  train the HandwrittenDigits machine learning model with MNIST dataset
  """
  @requirements ["app.start"]

  use Mix.Task

  alias HandwrittenDigits.Model
  alias HandwrittenDigits.MNISTDataset

  @impl Mix.Task
  def run(_args) do
    Mix.Shell.IO.info("Load MNIST dataset...")
    {train_images, train_labels} = Scidata.MNIST.download()
    {test_images, test_labels} = Scidata.MNIST.download_test()
    Mix.Shell.IO.info("Prepare MNIST datset...")
    batched_train_images = MNISTDataset.transform_images(train_images, 32)
    batched_train_labels = MNISTDataset.transform_labels(train_labels, 32)
    batched_test_images = MNISTDataset.transform_images(test_images, 32)
    batched_test_labels = MNISTDataset.transform_labels(test_labels, 32)
    data = Enum.zip(batched_train_images, batched_train_labels)
    testing_data = Enum.zip(batched_test_images, batched_test_labels)
    {training_data, validation_data} = Enum.split(data, floor(0.8 * Enum.count(data)))

    input_shape = {1, 28, 28}
    model = Model.build_model(input_shape)

    Mix.Shell.IO.info("Start training...")
    trained_model_state = Model.train(model, training_data, validation_data, [epochs: 10])

    # Mix.Shell.IO.info("Test model...")
    # Model.test(model, trained_model_state, testing_data)

    Mix.Shell.IO.info("Save model...")
    Model.save!(model, trained_model_state)
  end

end

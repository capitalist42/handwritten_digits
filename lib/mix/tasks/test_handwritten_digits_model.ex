defmodule Mix.Tasks.TestHandwrittenDigitsModel do
  @moduledoc """
  test the HandwrittenDigits machine learning model with MNIST dataset
  """
  @requirements ["app.start"]

  use Mix.Task

  alias HandwrittenDigits.Model
  alias HandwrittenDigits.MNISTDataset

  @impl Mix.Task
  def run(_args) do
    Mix.Shell.IO.info("\nLoad MNIST dataset...")
    {test_images, test_labels} = Scidata.MNIST.download_test()
    Mix.Shell.IO.info("\nPrepare MNIST datset...")
    batched_test_images = MNISTDataset.transform_images(test_images, 32)
    batched_test_labels = MNISTDataset.transform_labels(test_labels, 32)
    testing_data = Enum.zip(batched_test_images, batched_test_labels)

    input_shape = {1, 28, 28}
    model = Model.build_model(input_shape)

    Mix.Shell.IO.info("\nLoad trained model and model params...\n")
    {saved_model, saved_parms} = Model.load!()

    Mix.Shell.IO.info("\nTest model...\n")
    Model.test(saved_model, saved_parms, testing_data)

    Mix.Shell.IO.info("\nTesting finished!\n")

  end

end

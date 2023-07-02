defmodule HandwrittenDigits.ModelTest do
  use ExUnit.Case, async: false
  @checkpoints_dir_path Application.app_dir(:handwritten_digits, "priv/handwritten_digits_model/test_checkpoints")
  @model_params_path Application.app_dir(:handwritten_digits, "priv/handwritten_digits_model/test_model_params.axon")

  alias HandwrittenDigits.MNISTDataset
  alias HandwrittenDigits.Model

  describe "build_model/1" do
    test "returns a model" do
      input_shape = {1, 28, 28}
      model = Model.build_model(input_shape)
      assert map_size(model.nodes) == 6
      assert model.nodes[1].op == :input
      assert model.nodes[1].opts == [shape: {1, 28, 28}, optional: false]
      assert model.nodes[2].op == :flatten
      assert model.nodes[3].op == :dense
      assert model.nodes[4].op == :relu
      assert model.nodes[5].op == :dense
      assert model.nodes[6].op == :softmax
    end
  end

  describe "train/4" do
    test "train a model" do
      {train_images, train_labels} = Scidata.MNIST.download()
      {images_binary, images_type, images_shape} = train_images
      batched_train_images = MNISTDataset.transform_images(train_images, 32)
      batched_train_labels = MNISTDataset.transform_labels(train_labels, 32)
      data = Enum.zip(batched_train_images, batched_train_labels)
      {training_data, validation_data} = Enum.split(data, floor(0.8 * Enum.count(data)))

      model = Model.build_model({1, 28, 28})
      model_params = Model.train(model, training_data, validation_data, [epochs: 1, checkpoints_dir_path: @checkpoints_dir_path])
      {_init_fn, predict_fn} = Axon.build(model)
      input =
        images_binary
        |> Nx.from_binary(images_type)
        |> Nx.reshape(images_shape)
        |> Nx.divide(255)
        |> Nx.slice_along_axis(0, 1, axis: 0)
        |> Nx.reshape({1, 1, 28, 28})

      assert predict_fn.(model_params, input) |> Nx.argmax() |> Nx.to_number() == 5
    end
  end

  describe "save!/3" do
    test "saved model params to given path" do
      {train_images, train_labels} = Scidata.MNIST.download()
      batched_train_images = MNISTDataset.transform_images(train_images, 32)
      batched_train_labels = MNISTDataset.transform_labels(train_labels, 32)
      data = Enum.zip(batched_train_images, batched_train_labels)
      {training_data, validation_data} = Enum.split(data, floor(0.8 * Enum.count(data)))

      model = Model.build_model({1, 28, 28})
      model_params = Model.train(model, training_data, validation_data, [epochs: 1])
      Model.save!(model, model_params, path: @model_params_path)
      Model.load!(path: @model_params_path)
    end
  end

end

defmodule HandwrittenDigitsTest do
  use ExUnit.Case, async: true
  alias HandwrittenDigits.Model

  describe "serving model with Nx.Serving" do
    test "should handle four inputs and returns four predications (4 batch size)" do
      {test_images, test_labels} = Scidata.MNIST.download_test()
      {images_binary, images_type, images_shape} = test_images

      {saved_model, model_params} = Model.load!()
      {_init_fn, predict_fn} = Axon.build(saved_model)
      batch =
        images_binary
        |> Nx.from_binary(images_type)
        |> Nx.reshape(images_shape)
        |> Nx.divide(255)
        |> Nx.slice_along_axis(0, 4, axis: 0)
        |> Nx.to_batched(1)
        |> Enum.to_list()
        |> Nx.Batch.concatenate()

      batch |> IO.inspect(label:  "batch")
      # 7, 2, 1, 0
      predications = Nx.Serving.batched_run(HandwrittenDigits.Serving, batch)
      |> Nx.to_batched(1)
      |> Stream.map(fn(t) ->
        t |>  Nx.argmax() |> Nx.to_number()
      end)
      |> Enum.to_list()
      assert predications == [7, 2, 1, 0]
    end

    test "should handle list of one input and returns list of one predication" do
      {test_images, test_labels} = Scidata.MNIST.download_test()
      {images_binary, images_type, images_shape} = test_images

      {saved_model, model_params} = Model.load!()
      {_init_fn, predict_fn} = Axon.build(saved_model)
      input =
        images_binary
        |> Nx.from_binary(images_type)
        |> Nx.reshape(images_shape)
        |> Nx.divide(255)
        |> Nx.slice_along_axis(0, 1, axis: 0)
        |> IO.inspect()
      batch = Nx.Batch.concatenate([input])

      predications = Nx.Serving.batched_run(HandwrittenDigits.Serving, batch)
      |> Nx.to_batched(1)
      |> Stream.map(fn(t) ->
        t |>  Nx.argmax() |> Nx.to_number()
      end)
      |> Enum.to_list()
      assert predications == [7]
    end
  end
end

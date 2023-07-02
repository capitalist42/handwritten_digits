defmodule HandwrittenDigits.MNISTDataset do



  @doc """
    transform images from MNIST dataset to tensor (f32 type) with given batch size
  """
  @spec transform_images(tuple(), integer()) :: list(Nx.Tensor.t())
  def transform_images({images_binary, images_type, images_shape}, batch_size) do
    images_binary
    |> Nx.from_binary(images_type)
    |> Nx.reshape(images_shape)
    |> Nx.divide(255)
    |> Nx.to_batched(batch_size)
    |> Enum.to_list()
  end

  @doc """
    transform labels from MNIST dataset to tensor with given batch size
  """
  @spec transform_labels(tuple(), integer()) :: list(Nx.Tensor.t())
  def transform_labels({labels_binary, labels_type, _labels_shape}, batch_size) do
    categories_tensor = Nx.tensor(Enum.to_list(0..9))
    labels_binary
    |> Nx.from_binary(labels_type)
    |> Nx.new_axis(-1)
    |> Nx.equal(categories_tensor)
    |> Nx.to_batched(batch_size)
    |> Enum.to_list()
  end



end

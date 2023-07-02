defmodule HandwrittenDigits.MNISTDatasetTest do
  use ExUnit.Case, async: true
  alias HandwrittenDigits.MNISTDataset


  describe "transform_images/1" do
    test "convert images from MNIST dataset with batch size 100" do
      {train_images, _train_labels} = Scidata.MNIST.download()
      {images_binary, images_type, images_shape} = train_images
      batch_size = 100
      tensor_list = MNISTDataset.transform_images({images_binary, images_type, images_shape}, batch_size)
      assert length(tensor_list) == div(elem(images_shape, 0), batch_size)
      assert Nx.type(hd(tensor_list)) == {:f, 32}
      assert Nx.shape(hd(tensor_list)) == {100, 1, 28, 28}
      assert Nx.size(hd(tensor_list)) == batch_size * 1 * 28 * 28
    end
  end

  describe "transform_labels/1" do
    test "convert labels from MNIST dataset" do
      {_train_images, train_labels} = Scidata.MNIST.download()
      {labels_binary, labels_type, labels_shape} = train_labels
      {labels_count} = labels_shape
      batch_size = 100
      tensor_list = MNISTDataset.transform_labels({labels_binary, labels_type, labels_shape}, batch_size)
      assert length(tensor_list) == div(labels_count, batch_size)
      assert Nx.type(hd(tensor_list)) == {:u, 8}
      assert Nx.shape(hd(tensor_list)) == {batch_size, 10}
      assert Nx.size(hd(tensor_list)) == batch_size * 10
    end
  end
end

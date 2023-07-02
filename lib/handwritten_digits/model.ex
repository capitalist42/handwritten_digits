defmodule HandwrittenDigits.Model do
  @moduledoc """
  Machine Learning Model to detect handwritten digits, trainted with MNIST dataset.
  """

  @spec build_model(tuple()) :: Axon.t()
  def build_model(input_shape) do
    Axon.input("image", shape: input_shape)
    |> Axon.flatten()
    |> Axon.dense(784, activation: :relu)
    |> Axon.dense(10, activation: :softmax)
  end

  def train(model, training_data, validation_data, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 10)
    checkpoints_dir = Keyword.get(opts, :checkpoints_dir_path, get_checkpoints_dir_path())

    loop =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
      |> Axon.Loop.metric(:accuracy, "Accuracy")
      |> Axon.Loop.metric(:precision, "Precision")
      |> Axon.Loop.validate(model, validation_data)

    case load_model_state_from_checkpoint(checkpoints_dir) do
      {:ok, nil} ->
        loop
        |> Axon.Loop.checkpoint(path: checkpoints_dir, criteria: "Accuracy", mode: :max)
        |> Axon.Loop.run(training_data, %{}, epochs: epochs, compiler: EXLA)

      {:ok, state} ->
        loop
        |> Axon.Loop.from_state(state)
        |> Axon.Loop.run(training_data, %{}, epochs: epochs, compiler: EXLA)
    end
  end

  # def test(model, trained_model_state, testing_data) do
  #   model
  #   |> Axon.Loop.evaluator(model)
  #   |> Axon.Loop.metric(:mean_absolute_error)
  #   |> Axon.Loop.run(testing_data, trained_model_state, iterations: 1000, compiler: EXLA)
  # end

  def save!(model, params, opts \\ []) do
    model_params_path = Keyword.get(opts, :path, get_model_params_path())
    contents = Axon.serialize(model, params)
    File.write!(model_params_path, contents)
  end

  def load!(opts \\ []) do
    model_params_path = Keyword.get(opts, :path, get_model_params_path())
    contents = File.read!(model_params_path)
    Axon.deserialize(contents)
  end

  defp get_checkpoints_dir_path(),
    do: Application.app_dir(:handwritten_digits, "priv/handwritten_digits_model/checkpoints")

  defp get_model_params_path(),
    do: Application.app_dir(:handwritten_digits, "priv/handwritten_digits_model/model_params.axon")

  defp load_model_state_from_checkpoint(dir) do
    with {:ok, checkpoint_file_path} <- find_latest_checkpoint_file(dir) do
      IO.puts("Found checkpoint file: #{checkpoint_file_path}")

      state =
        checkpoint_file_path
        |> File.read!()
        |> Axon.Loop.deserialize_state()

      {:ok, state}
    else
      {:error, :no_checkpoint_found} ->
        IO.puts("No checkpoint file found in #{dir}")
        {:ok, nil}
    end
  end

  defp find_latest_checkpoint_file(dir) do
    case File.exists?(dir) do
      true ->
        maybe_checkpoint_file =
          dir
          |> File.ls!()
          |> Enum.filter(&(&1 =~ ~r/\.ckpt$/))
          |> Enum.map(fn file ->
            path = Path.join(dir, file)
            stat = path |> File.stat!()
            {path, stat}
          end)
          |> Enum.sort_by(fn {_path, stat} -> stat.mtime end, :desc)
          |> Enum.at(0)

        case maybe_checkpoint_file do
          nil -> {:error, :no_checkpoint_found}
          {checkpoint_file_path, _stat} -> {:ok, checkpoint_file_path}
        end

      false ->
        {:error, :no_checkpoint_found}
    end
  end
end

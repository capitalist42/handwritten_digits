defmodule HandwrittenDigits.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Start the Telemetry supervisor
      HandwrittenDigitsWeb.Telemetry,
      # Start the PubSub system
      {Phoenix.PubSub, name: HandwrittenDigits.PubSub},
      # Start Finch
      {Finch, name: HandwrittenDigits.Finch},
      # Start the Endpoint (http/https)
      HandwrittenDigitsWeb.Endpoint,
      # Start a worker by calling: HandwrittenDigits.Worker.start_link(arg)
      # {HandwrittenDigits.Worker, arg}
      {Nx.Serving, serving: build_serving(), name: HandwrittenDigits.Serving, batch_timeout: 100}
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: HandwrittenDigits.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    HandwrittenDigitsWeb.Endpoint.config_change(changed, removed)
    :ok
  end

  def build_serving() do
    # Configuration
    batch_size = 4
    build_model_options = [compiler: EXLA, mode: :inference]
    defn_options = [compiler: EXLA]

    Nx.Serving.new(
      fn ->
        {saved_model, saved_params} = HandwrittenDigits.Model.load!()

        # Build the prediction defn function
        {_init_fun, predict_fun} = Axon.build(saved_model, build_model_options)

        inputs_template = %{"pixel_values" => Nx.template({batch_size, 1, 28, 28}, :f32)}
        template_args = [Nx.to_template(saved_params), inputs_template]

        # Compile the prediction function upfront for the configured batch_size
        predict_fun = Nx.Defn.compile(predict_fun, template_args, defn_options)

        # The returned function is called for every accumulated batch
        fn inputs ->
          inputs = Nx.Batch.pad(inputs, batch_size - inputs.size)
          predict_fun.(saved_params, inputs)
        end
      end,
      batch_size: batch_size
    )
  end
end

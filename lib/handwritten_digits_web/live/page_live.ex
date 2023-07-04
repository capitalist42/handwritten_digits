defmodule HandwrittenDigitsWeb.PageLive do
  use HandwrittenDigitsWeb, :live_view

  def render(assigns) do
    assigns |> IO.inspect(label: "assigns")

    ~H"""
    <div id="wrapper" phx-update="ignore">
      <div id="canvas" phx-hook="Draw"></div>
    </div>

    <div>
      <button phx-click="reset-button-clicked">Reset</button>
      <button phx-click="predict-button-clicked">Predict</button>
    </div>

    <%= if @prediction do %>
      <div>
        <div>
          Prediction:
        </div>
        <div>
          <%= @prediction %>
        </div>
      </div>
    <% end %>
    """
  end

  def mount(params, session, socket) do
    params |> IO.inspect(label: "params")
    session |> IO.inspect(label: "session")
    socket |> IO.inspect(label: "socket")
    {:ok, assign(socket, %{prediction: nil})}
  end

  def handle_event("reset-button-clicked", _params, socket) do
    {:noreply,
     socket
     |> assign(prediction: nil)
     |> push_event("reset-button-clicked", %{})}
  end

  def handle_event("predict-button-clicked", _params, socket) do
    socket = socket |> push_event("predict-button-clicked", %{})
    {:noreply, socket}
  end

  def handle_event("canvas-image-data-captured", %{"image" => image}, socket) do
    "data:image/png;base64," <> image_raw = image
    name = Base.url_encode64(:crypto.strong_rand_bytes(10), padding: false)
    path = Path.join(System.tmp_dir!(), "#{name}.webp")
    path |> IO.inspect(label: "path")
    File.write!(path, Base.decode64!(image_raw))

    mat =
      Evision.imread(path, flags: Evision.Constant.cv_IMREAD_GRAYSCALE())
      |> Evision.resize({28, 28})

    data =
      Evision.Mat.to_nx(mat, EXLA.Backend)
      |> Nx.as_type(:f32)
      |> Nx.reshape({1, 28, 28})
      |> List.wrap()
      |> IO.inspect(label: "List.wrap/1")

    batch = Nx.Batch.stack(data)
      |> IO.inspect(label: "Nx.Batch.stack/1")

    prediction =
      Nx.Serving.batched_run(HandwrittenDigits.Serving, batch)
      |> Nx.to_batched(1)
      |> Enum.to_list()
      |> hd()
      |> Nx.argmax()
      |> Nx.to_number()
    socket = socket |> assign(:prediction, prediction)

    {:noreply, socket}
  end
end

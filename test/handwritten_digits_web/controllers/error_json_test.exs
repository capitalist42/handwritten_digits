defmodule HandwrittenDigitsWeb.ErrorJSONTest do
  use HandwrittenDigitsWeb.ConnCase, async: true

  test "renders 404" do
    assert HandwrittenDigitsWeb.ErrorJSON.render("404.json", %{}) == %{errors: %{detail: "Not Found"}}
  end

  test "renders 500" do
    assert HandwrittenDigitsWeb.ErrorJSON.render("500.json", %{}) ==
             %{errors: %{detail: "Internal Server Error"}}
  end
end

import Config

# We don't run a server during test. If one is required,
# you can enable the server option below.
config :handwritten_digits, HandwrittenDigitsWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "/HVlMckvKiIWV8MFMxn0yUFoBzYWMD1ICIeremaNx9ZbFsZV8b4+rDT2CPUv12Hl",
  server: false

# Print only warnings and errors during test
config :logger, level: :warning

# Initialize plugs at runtime for faster test compilation
config :phoenix, :plug_init_mode, :runtime

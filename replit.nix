{ pkgs }: {
  deps = [
    pkgs.busybox
    pkgs.inotify-tools
    pkgs.elixir_1_14
  ];
}
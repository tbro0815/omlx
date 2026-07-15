require_relative "omlx"

# Local Homebrew formula for the consolidated oMLX release.  It inherits the
# official formula's dependency and build rules, changing only the source and
# release identity.
class OmlxOmni < Omlx
  desc "oMLX with Jang and external-model support"
  homepage "https://github.com/tbro0815/omlx"
  url "https://github.com/tbro0815/omlx/archive/refs/tags/v0.5.1-omni.tar.gz"
  version "0.5.1-omni"
  sha256 "b51a8d64b5dab4c80349918f9e16822bff1306b99eb6ea2188d3a7e2ab7dee1a"

  # Homebrew does not carry dependencies from a parent formula into a
  # separately-installed child formula.
  depends_on "rust" => :build
  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.11"
end

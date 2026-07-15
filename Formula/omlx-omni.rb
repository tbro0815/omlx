require_relative "omlx"

# Local Homebrew formula for the consolidated oMLX release.  It inherits the
# official formula's dependency and build rules, changing only the source and
# release identity.
class OmlxOmni < Omlx
  desc "oMLX with Jang and external-model support"
  homepage "https://github.com/tbro0815/omlx"
  url "https://github.com/tbro0815/omlx/archive/refs/tags/v0.5.1-omni.tar.gz"
  version "0.5.1-omni"
  sha256 "387d4f2a2386dc76ebacb89b70874122513aa4cb5e098ba9ad602881585cdc74"

  # Homebrew does not carry dependencies from a parent formula into a
  # separately-installed child formula.
  depends_on "rust" => :build
  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.11"
end

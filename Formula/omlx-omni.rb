require_relative "omlx"

# Local Homebrew formula for the consolidated oMLX release.  It inherits the
# official formula's dependency and build rules, changing only the source and
# release identity.
class OmlxOmni < Omlx
  desc "oMLX with Jang and external-model support"
  homepage "https://github.com/tbro0815/omlx"
  url "https://github.com/tbro0815/omlx/archive/refs/tags/v0.5.3-omni.tar.gz"
  version "0.5.3-omni"
  sha256 "24bbf4b4a4292f5c9f379d716a8fdb7d606c18ec9a38c2a34963fc62db327f05"

  # Homebrew does not carry dependencies from a parent formula into a
  # separately-installed child formula.
  depends_on "rust" => :build
  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.11"
end

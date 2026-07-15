require_relative "omlx"

# Local Homebrew formula for the consolidated oMLX release.  It inherits the
# official formula's dependency and build rules, changing only the source and
# release identity.
class OmlxOmni < Omlx
  desc "oMLX with Jang and external-model support"
  homepage "https://github.com/tbro0815/omlx"
  url "https://github.com/tbro0815/omlx/archive/refs/tags/v0.5.1-omni.tar.gz"
  version "0.5.1-omni"
  sha256 "d82271d8445f10b9da8464ca44c2284494ae0a9d22ada0c8d0656f167fa73e28"

  # Homebrew does not carry dependencies from a parent formula into a
  # separately-installed child formula.
  depends_on "rust" => :build
  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.11"
end

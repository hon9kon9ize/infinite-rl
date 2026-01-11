from setuptools import setup
from setuptools.command.install import install
import subprocess
import platform


class CustomInstall(install):
    def run(self):
        # Trigger the system-level installations
        print("Installing multi-language runtimes for coding tasks...")

        system = platform.system()

        if system == "Darwin":  # macOS
            self._install_macos()
        elif system == "Linux":
            self._install_linux()
        elif system == "Windows":
            self._install_windows()
        else:
            print(
                f"Warning: Unsupported OS {system}. Some runtimes may not be available."
            )

        install.run(self)

    def _install_macos(self):
        """Install dependencies on macOS using Homebrew"""
        print("Installing dependencies on macOS...")

        # Check if Homebrew is installed
        try:
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Homebrew not found. Please install Homebrew from https://brew.sh")
            return

        packages = ["node", "openjdk@17", "gcc", "rustup"]
        for package in packages:
            try:
                print(f"Installing {package}...")
                subprocess.run(["brew", "install", package], check=False)
            except Exception as e:
                print(f"Warning: Failed to install {package}: {e}")

        # Install ts-node for TypeScript execution
        try:
            print("Installing ts-node...")
            subprocess.run(["npm", "install", "-g", "ts-node"], check=False)
        except Exception as e:
            print(f"Warning: Failed to install ts-node: {e}")

    def _install_linux(self):
        """Install dependencies on Linux using apt-get and NodeSource"""
        print("Installing dependencies on Linux...")
        try:
            # Install NodeSource for a modern Node.js version (Node 20+)
            # Colab's default nodejs is often too old (v12) which causes SyntaxErrors with modern TS
            print("Configuring NodeSource for Node.js 20...")
            subprocess.run(
                "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
                shell=True,
                check=False,
            )
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(
                ["apt-get", "install", "-y", "nodejs", "openjdk-17-jdk", "g++"],
                check=True,
            )
        except Exception as e:
            print(f"Warning: Failed to install via apt-get or NodeSource: {e}")
            # Fallback to standard apt-get if NodeSource fails
            try:
                subprocess.run(
                    [
                        "apt-get",
                        "install",
                        "-y",
                        "nodejs",
                        "npm",
                        "openjdk-17-jdk",
                        "g++",
                    ],
                    check=False,
                )
            except Exception:
                pass

        # Install Rust
        try:
            print("Installing Rust...")
            subprocess.run(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                shell=True,
            )
        except Exception as e:
            print(f"Warning: Failed to install Rust: {e}")

        # Install ts-node for TypeScript execution
        try:
            print("Installing ts-node...")
            subprocess.run(["npm", "install", "-g", "ts-node"], check=False)
        except Exception as e:
            print(f"Warning: Failed to install ts-node: {e}")

    def _install_windows(self):
        """Install dependencies on Windows"""
        print("Installing dependencies on Windows...")
        print("Note: Manual installation may be required for some tools.")
        print("- Node.js: Download from https://nodejs.org")
        print(
            "- Java 17: Download from https://www.oracle.com/java/technologies/downloads"
        )
        print("- g++: Install via MinGW or Visual Studio")
        print("- Rust: Download from https://rustup.rs")

        # Try to install ts-node if npm is available
        try:
            print("Installing ts-node...")
            subprocess.run(["npm", "install", "-g", "ts-node"], check=False)
        except Exception as e:
            print(
                f"Warning: Failed to install ts-node. Install it manually with: npm install -g ts-node"
            )


setup(
    name="infinite_rl",
    version="0.1.3",
    packages=["infinite_rl", "infinite_rl.reward_functions", "infinite_rl.examples"],
    include_package_data=True,
    package_data={
        "infinite_rl.examples": ["*.md"],
        "infinite_rl.reward_functions": ["*.py"],
    },
    cmdclass={"install": CustomInstall},
    install_requires=[
        "requests",
        "google-generativeai",
        "pandas",
        "python-dotenv",
        "sympy",
        "antlr4-python3-runtime==4.11.1",
        "sentence-transformers",
        "transformers",
        "bs4",
        "gguf",
    ],
)

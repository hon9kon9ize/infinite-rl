from setuptools import setup
from setuptools.command.install import install
import subprocess
import platform
import os


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

        # Helper to run commands with sudo if available
        def run_cmd(cmd, shell=False):
            if isinstance(cmd, str) and not shell:
                cmd = cmd.split()

            # Try with sudo first if it's linux and not root
            if os.geteuid() != 0:
                if shell:
                    cmd = f"sudo {cmd}"
                else:
                    cmd = ["sudo"] + cmd

            return subprocess.run(
                cmd, shell=shell, check=False, capture_output=True, text=True
            )

        try:
            # Check current node version
            node_v_proc = run_cmd(["node", "-v"])
            if node_v_proc.returncode == 0:
                node_v = node_v_proc.stdout.strip().replace("v", "")
                major_v = int(node_v.split(".")[0]) if node_v else 0
                if major_v >= 18:
                    print(f"Node.js version {node_v} is already modern enough.")
                else:
                    raise ValueError("Node too old")
            else:
                raise FileNotFoundError()
        except (ValueError, FileNotFoundError):
            # Install NodeSource for a modern Node.js version (Node 20+)
            print("Configuring NodeSource for Node.js 20...")
            run_cmd(
                "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -", shell=True
            )
            run_cmd(["apt-get", "update"])
            # Purge old versions first to be safe
            run_cmd(["apt-get", "remove", "-y", "nodejs", "npm"])
            run_cmd(["apt-get", "install", "-y", "nodejs", "openjdk-17-jdk", "g++"])

        # Install ts-node and typescript globally
        print("Installing ts-node and typescript globally...")
        run_cmd(["npm", "install", "-g", "ts-node", "typescript"])

        # Install Rust
        try:
            # Check if rustc is available
            subprocess.run(["rustc", "--version"], check=True, capture_output=True)
            print("Rust is already installed.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Installing Rust...")
            subprocess.run(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                shell=True,
            )
        except Exception as e:
            print(f"Warning: Failed to install Rust: {e}")

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
    version="0.1.7",
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

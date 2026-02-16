import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Network_proxy:
    clash_proxy = "http://127.0.0.1:7897"

    @staticmethod
    def open_agent(op: str = "clash"):
        os.environ["http_proxy"] = Network_proxy.clash_proxy
        os.environ["https_proxy"] = Network_proxy.clash_proxy

    @staticmethod
    def close_agent():
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
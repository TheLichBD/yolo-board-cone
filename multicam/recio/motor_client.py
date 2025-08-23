# multicam/io/motor_client.py
import socket

class MotorClient:
    def __init__(self, host: str, port: int = 5000, timeout: float = 0.2):
        self.host = host
        self.port = port
        self.timeout = timeout

    def send_error(self, err_x: float, err_y: float) -> bool:
        msg = f"{err_x},{err_y}".encode("utf-8")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.host, self.port))
                s.sendall(msg)
                resp = s.recv(1024)
                return resp == b"OK"
        except Exception:
            return False

from flask import Flask
import os

app = Flask(__name__)
def create_app():
    
    # Đăng ký các routes từ Blueprint
    from .routes import main
    app.register_blueprint(main)
    # In thông tin về thư mục template
    print("Current template folder:", app.template_folder)
    print("Files in template folder:", os.listdir(app.template_folder))

    return app

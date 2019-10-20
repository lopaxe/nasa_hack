class AppConfig:

    def __init__(self):
        self.UPLOAD_PATH="uploaded_files"
        self.STYLE_PATH="pics/styles"
        self.TRANSFORMATION_PATH="pics/transformations"
        self.TRANSFORMATION_PATH = "pics/transformations"

    @property
    def fire_path(self):
        return f"{self.STYLE_PATH}/fire/fire.jpeg"

    @property
    def ice_path(self):
        return f"{self.STYLE_PATH}/ice/ice.jpg"

    @property
    def original_transformation_path(self):
        return f"{self.TRANSFORMATION_PATH}/original"

    @property
    def full_transformation_path(self):
        return f"{self.TRANSFORMATION_PATH}/full_transformation"

    @property
    def weigted_transformation_path(self):
        return f"{self.TRANSFORMATION_PATH}/weighted_transformation"

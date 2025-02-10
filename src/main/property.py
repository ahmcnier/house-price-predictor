class Property:
    def __init__(self, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus):
        self.area = area
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.stories = stories
        self.mainroad = mainroad
        self.guestroom = guestroom
        self.basement = basement
        self.hotwaterheating = hotwaterheating
        self.airconditioning = airconditioning
        self.parking = parking
        self.prefarea = prefarea
        self.furnishingstatus = furnishingstatus

    def to_array(self):
        return [[self.area, self.bedrooms, self.bathrooms, self.stories, self.mainroad, self.guestroom, self.basement, self.hotwaterheating, self.airconditioning, self.parking, self.prefarea, self.furnishingstatus]]
CREATE DATABASE smartev1;
USE smartev1;
CREATE TABLE users2(
user_id INT PRIMARY KEY AUTO_INCREMENT ,
full_name VARCHAR(100) NOT NULL,
email VARCHAR(100) UNIQUE,
phone VARCHAR(15),
city VARCHAR(50),
registration_date DATE 
);

CREATE TABLE Charging_Stations(
station_id INT PRIMARY KEY AUTO_INCREMENT,
station_name VARCHAR(100),
location VARCHAR(100),
total_ports INT, 
fast_charging BOOLEAN
);

CREATE TABLE vehicals(
vehical_id INT PRIMARY KEY AUTO_INCREMENT,
user_id INT,
vehical_model
VARCHAR(100),
battery_capacity_kwh DECIMAL(5.2),
FOREIGN KEY(user_id)
REFERENCES users(user_id)
);

CREATE TABLE Charging_Sessions (
    session_id INT PRIMARY KEY AUTO_INCREMENT,
    vehical_id INT,
    station_id INT,
    start_time DATETIME,
    end_time DATETIME,
    units_consumed DECIMAL(6,2),
    total_cost DECIMAL(8,2),
    FOREIGN KEY (vehical_id) REFERENCES vehicals(vehical_id),
    FOREIGN KEY (station_id) REFERENCES Charging_Stations(station_id)
);


INSERT INTO Users1(full_name, email, phone, city) VALUES
('Rohan Mehta', 'rohan@gmail.com', '9876543210', 'Bangalore'),
('Ananya Iyer', 'ananya@gmail.com', '9123456780', 'Chennai'),
('Vikram Rao', 'vikram@gmail.com', '9988776655', 'Hyderabad'),
('Neha Sharma', 'neha@gmail.com', '9001122334', 'Mumbai');

INSERT INTO Charging_Stations (station_name, location, total_ports, fast_charging) VALUES
('GreenCharge Hub', 'Indiranagar', 8, TRUE),
('VoltPoint Station', 'Koramangala', 6, TRUE),
('EcoPlug Center', 'Whitefield', 4, FALSE);
DESCRIBE vehicals;
INSERT INTO vehicals (user2_id, vehical_model, battery_capacity_kwh) VALUES
(1, 'Tata Nexon EV', 40.50),
(2, 'MG ZS EV', 44.50),
(3, 'Hyundai Kona Electric', 39.20),
(1, 'Mahindra XUV400', 34.50);

INSERT INTO Charging_Sessions 
(vehical_id, station_id, start_time, end_time, units_consumed, total_cost)
VALUES
(1, 1, '2024-03-01 10:00:00', '2024-03-01 11:30:00', 18.5, 370.00),
(2, 2, '2024-03-02 09:00:00', '2024-03-02 10:15:00', 15.0, 300.00),
(3, 1, '2024-03-03 14:00:00', '2024-03-03 15:00:00', 12.0, 240.00),
(1, 3, '2024-03-04 18:00:00', '2024-03-04 19:45:00', 20.0, 400.00);


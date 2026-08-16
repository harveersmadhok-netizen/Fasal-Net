# ============================================================
# FASAL-NET
# Python backend for the HTML frontend
# ============================================================

import math
import os
import re
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ------------------------------------------------------------
# File locations
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATABASE_FILE = BASE_DIR / "fasal_net.db"

MODEL_FILE = BASE_DIR / "fasal_price_model.pkl"

FRONTEND_FILE = (
    BASE_DIR
    / "static"
    / "index.html"
)


# ------------------------------------------------------------
# Bigha conversion
# ------------------------------------------------------------
#
# Bigha is different in different parts of India.
# This value can be changed later for your chosen region.

BIGHA_TO_ACRE = float(
    os.getenv(
        "BIGHA_TO_ACRE",
        "0.625"
    )
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Fasal-Net",
    version="1.0"
)


# Lets the frontend talk to Python during testing

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# ERROR MESSAGES
# ============================================================
#
# Your HTML looks for:
#
# data.message
#
# so Python sends errors in that format.


@app.exception_handler(HTTPException)
async def http_error(
    request: Request,
    exc: HTTPException
):

    if isinstance(
        exc.detail,
        dict
    ):

        message = (
            exc.detail.get("message")
            or str(exc.detail)
        )

    else:

        message = str(
            exc.detail
        )


    return JSONResponse(
        status_code=exc.status_code,

        content={
            "message": message
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_error(
    request: Request,
    exc: RequestValidationError
):

    errors = exc.errors()


    if errors:

        error = errors[0]

        location = error.get(
            "loc",
            []
        )


        field_name = (

            str(location[-1])

            if location

            else "input"

        )


        message = (
            f"Please check {field_name}"
        )

    else:

        message = (
            "Please check the entered details"
        )


    return JSONResponse(
        status_code=422,

        content={
            "message": message
        }
    )


@app.exception_handler(Exception)
async def server_error(
    request: Request,
    exc: Exception
):

    # This prints the real error in the terminal
    # while showing a simpler message in the app.

    print(
        "Server error:",
        repr(exc)
    )


    return JSONResponse(
        status_code=500,

        content={
            "message":
                "Something went wrong on the server"
        }
    )


# ============================================================
# DATABASE
# ============================================================


def get_connection():

    connection = sqlite3.connect(
        str(DATABASE_FILE)
    )


    connection.row_factory = (
        sqlite3.Row
    )


    connection.execute(
        "PRAGMA foreign_keys = ON"
    )


    return connection


def create_database():

    connection = get_connection()

    cursor = connection.cursor()


    # --------------------------------------------------------
    # Farmers
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS farmers (

            farmer_id INTEGER
                PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            contact TEXT NOT NULL,

            location_text TEXT NOT NULL,

            latitude REAL,

            longitude REAL,

            created_at TEXT
                DEFAULT CURRENT_TIMESTAMP

        )
        """
    )


    # --------------------------------------------------------
    # Fields
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS fields (

            field_id INTEGER
                PRIMARY KEY AUTOINCREMENT,

            farmer_id INTEGER NOT NULL,

            location_text TEXT NOT NULL,

            latitude REAL,

            longitude REAL,

            entered_size REAL NOT NULL,

            entered_unit TEXT NOT NULL,

            size_acres REAL,

            size_hectares REAL,

            created_at TEXT
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (farmer_id)
            REFERENCES farmers(farmer_id)
            ON DELETE CASCADE

        )
        """
    )


    # --------------------------------------------------------
    # Farmer crops
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS farmer_crops (

            farmer_crop_id INTEGER
                PRIMARY KEY AUTOINCREMENT,

            farmer_id INTEGER NOT NULL,

            field_id INTEGER NOT NULL,

            crop_name TEXT NOT NULL,

            entered_quantity REAL NOT NULL,

            entered_unit TEXT NOT NULL,

            quantity_kg REAL NOT NULL,

            quantity_quintal REAL NOT NULL,

            quantity_tonne REAL NOT NULL,

            created_at TEXT
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (farmer_id)
            REFERENCES farmers(farmer_id)
            ON DELETE CASCADE,

            FOREIGN KEY (field_id)
            REFERENCES fields(field_id)
            ON DELETE CASCADE

        )
        """
    )


    # --------------------------------------------------------
    # Transporters
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transporters (

            transporter_id INTEGER
                PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            contact TEXT NOT NULL,

            location_text TEXT NOT NULL,

            latitude REAL,

            longitude REAL,

            min_price_per_tonne REAL
                NOT NULL,

            currency TEXT
                DEFAULT 'INR',

            verified INTEGER
                DEFAULT 0,

            active INTEGER
                DEFAULT 1,

            created_at TEXT
                DEFAULT CURRENT_TIMESTAMP

        )
        """
    )


    # --------------------------------------------------------
    # Markets
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS markets (

            market_id INTEGER
                PRIMARY KEY AUTOINCREMENT,

            market_name TEXT NOT NULL,

            city TEXT,

            district TEXT,

            state TEXT,

            latitude REAL NOT NULL,

            longitude REAL NOT NULL,

            active INTEGER
                DEFAULT 1

        )
        """
    )


    # --------------------------------------------------------
    # Old crop prices
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_prices (

            price_id INTEGER
                PRIMARY KEY AUTOINCREMENT,

            market_id INTEGER NOT NULL,

            crop_name TEXT NOT NULL,

            price_date TEXT NOT NULL,

            month INTEGER NOT NULL,

            year INTEGER NOT NULL,

            minimum_price REAL,

            maximum_price REAL,

            modal_price REAL NOT NULL,

            previous_price REAL,

            supply_quantity REAL
                DEFAULT 0,

            demand_index REAL
                DEFAULT 1,

            FOREIGN KEY (market_id)
            REFERENCES markets(market_id)
            ON DELETE CASCADE

        )
        """
    )


    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS storage_facilities (

            storage_id INTEGER
                PRIMARY KEY AUTOINCREMENT,

            market_id INTEGER NOT NULL,

            storage_name TEXT NOT NULL,

            available_capacity_quintal REAL
                NOT NULL,

            cost_per_quintal_per_day REAL
                NOT NULL,

            contact TEXT,

            active INTEGER
                DEFAULT 1,

            FOREIGN KEY (market_id)
            REFERENCES markets(market_id)
            ON DELETE CASCADE

        )
        """
    )


    # --------------------------------------------------------
    # Market recommendations
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations (

            recommendation_id INTEGER
                PRIMARY KEY AUTOINCREMENT,

            farmer_id INTEGER NOT NULL,

            crop_name TEXT NOT NULL,

            market_id INTEGER NOT NULL,

            transporter_id INTEGER,

            predicted_price REAL NOT NULL,

            gross_value REAL NOT NULL,

            transport_cost REAL NOT NULL,

            storage_cost REAL NOT NULL,

            total_cost REAL NOT NULL,

            expected_amount REAL NOT NULL,

            distance_km REAL,

            created_at TEXT
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (farmer_id)
            REFERENCES farmers(farmer_id),

            FOREIGN KEY (market_id)
            REFERENCES markets(market_id),

            FOREIGN KEY (transporter_id)
            REFERENCES transporters(transporter_id)

        )
        """
    )


    # --------------------------------------------------------
    # Farmer and transporter deal
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (

            transaction_id INTEGER
                PRIMARY KEY AUTOINCREMENT,

            farmer_id INTEGER NOT NULL,

            transporter_id INTEGER NOT NULL,

            market_id INTEGER,

            crop_name TEXT NOT NULL,

            quantity_kg REAL NOT NULL,

            transport_price REAL NOT NULL,

            status TEXT
                DEFAULT 'CREATED',

            created_at TEXT
                DEFAULT CURRENT_TIMESTAMP,

            completed_at TEXT,

            FOREIGN KEY (farmer_id)
            REFERENCES farmers(farmer_id),

            FOREIGN KEY (transporter_id)
            REFERENCES transporters(transporter_id),

            FOREIGN KEY (market_id)
            REFERENCES markets(market_id)

        )
        """
    )


    # --------------------------------------------------------
    # Shipment
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS shipments (

            shipment_id INTEGER
                PRIMARY KEY AUTOINCREMENT,

            transaction_id INTEGER NOT NULL,

            current_status TEXT
                DEFAULT 'TRANSPORT BOOKED',

            last_updated TEXT
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (transaction_id)
            REFERENCES transactions(transaction_id)
            ON DELETE CASCADE

        )
        """
    )


    # --------------------------------------------------------
    # Shipment history
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS shipment_updates (

            update_id INTEGER
                PRIMARY KEY AUTOINCREMENT,

            shipment_id INTEGER NOT NULL,

            status TEXT NOT NULL,

            message TEXT,

            latitude REAL,

            longitude REAL,

            update_time TEXT
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (shipment_id)
            REFERENCES shipments(shipment_id)
            ON DELETE CASCADE

        )
        """
    )


    connection.commit()

    connection.close()


# Make the database when the app starts

create_database()


# ============================================================
# FRONTEND INPUT SHAPES
# ============================================================


class LocationInput(BaseModel):

    text: str

    lat: Optional[float] = None

    lng: Optional[float] = None


class FieldSizeInput(BaseModel):

    value: float = Field(
        gt=0
    )

    unit: str


class CropInput(BaseModel):

    crop: str

    quantity: float = Field(
        gt=0
    )

    unit: str


class FarmerRegistration(BaseModel):

    name: str

    contact: str

    location: LocationInput

    fieldSize: FieldSizeInput

    crops: List[CropInput]


class TransporterRegistration(BaseModel):

    name: str

    contact: str

    location: LocationInput

    minPricePerTonne: float = Field(
        gt=0
    )

    currency: str = "INR"


# ============================================================
# OTHER INPUT SHAPES
# ============================================================


class MarketInput(BaseModel):

    marketName: str

    city: str

    district: str

    state: str

    lat: float

    lng: float


class PriceInput(BaseModel):

    marketId: int

    crop: str

    date: str

    minimumPrice: Optional[float] = None

    maximumPrice: Optional[float] = None

    modalPrice: float = Field(
        gt=0
    )

    previousPrice: Optional[float] = None

    supplyQuantity: float = 0

    demandIndex: float = 1


class StorageInput(BaseModel):

    marketId: int

    name: str

    availableCapacityQuintal: float = Field(
        gt=0
    )

    costPerQuintalPerDay: float = Field(
        gt=0
    )

    contact: Optional[str] = None


class RecommendationInput(BaseModel):

    farmerId: int

    crop: str

    quantity: float = Field(
        gt=0
    )

    unit: str

    sellingMonth: int

    storageDays: int = 0

    maximumDistanceKm: float = 250


class TransactionInput(BaseModel):

    farmerId: int

    transporterId: int

    marketId: Optional[int] = None

    crop: str

    quantity: float = Field(
        gt=0
    )

    unit: str


class ShipmentUpdateInput(BaseModel):

    status: str

    message: Optional[str] = None

    lat: Optional[float] = None

    lng: Optional[float] = None


# ============================================================
# BASIC CHECKS
# ============================================================


def check_phone(
    phone
):

    # Remove spaces because the frontend placeholder
    # shows phone numbers with a space.

    phone = re.sub(
        r"\s+",
        "",
        str(phone)
    )


    if not re.fullmatch(
        r"[6-9][0-9]{9}",
        phone
    ):

        raise ValueError(
            "Enter a valid 10-digit mobile number"
        )


    return phone


def check_location(
    location
):

    if not location.text.strip():

        raise ValueError(
            "Location is required"
        )


    # GPS is optional when registering.

    if (
        location.lat is not None
        or location.lng is not None
    ):

        if (
            location.lat is None
            or location.lng is None
        ):

            raise ValueError(
                "Both latitude and longitude are needed"
            )


        if not -90 <= location.lat <= 90:

            raise ValueError(
                "Latitude is not valid"
            )


        if not -180 <= location.lng <= 180:

            raise ValueError(
                "Longitude is not valid"
            )


def check_month(
    month
):

    if month < 1 or month > 12:

        raise ValueError(
            "Month must be between 1 and 12"
        )


# ============================================================
# CROP UNIT CONVERSION
# ============================================================


def convert_crop_to_kg(
    quantity,
    unit
):

    unit = (
        unit
        .strip()
        .lower()
    )


    conversions = {

        "kg":
            1,

        "quintal":
            100,

        "tonne":
            1000,

        # Your HTML says:
        # Bag (50 kg)

        "bag50":
            50

    }


    if unit not in conversions:

        raise ValueError(
            "Crop unit must be kg, quintal, tonne or bag50"
        )


    kg = (

        quantity

        * conversions[
            unit
        ]

    )


    return round(
        kg,
        2
    )


def standardize_crop_quantity(
    quantity,
    unit
):

    kg = convert_crop_to_kg(
        quantity,
        unit
    )


    return {

        "kg":
            round(
                kg,
                2
            ),

        "quintal":
            round(
                kg / 100,
                2
            ),

        "tonne":
            round(
                kg / 1000,
                3
            )

    }


# ============================================================
# FIELD SIZE CONVERSION
# ============================================================


def standardize_field_size(
    value,
    unit
):

    unit = (
        unit
        .strip()
        .lower()
    )


    if value <= 0:

        raise ValueError(
            "Field size must be more than 0"
        )


    if unit == "acre":

        acres = value


    elif unit == "hectare":

        acres = (

            value

            * 2.47105381

        )


    elif unit == "bigha":

        acres = (

            value

            * BIGHA_TO_ACRE

        )


    else:

        raise ValueError(
            "Field unit must be acre, hectare or bigha"
        )


    hectares = (

        acres

        * 0.4046856422

    )


    return {

        "acres":
            round(
                acres,
                3
            ),

        "hectares":
            round(
                hectares,
                3
            )

    }


# ============================================================
# FARMER REGISTRATION
# ============================================================
#
# This endpoint matches your HTML:
#
# POST /api/farmers/register
# ============================================================


@app.post(
    "/api/farmers/register"
)
def register_farmer(
    data: FarmerRegistration
):

    try:

        if not data.name.strip():

            raise ValueError(
                "Enter your name"
            )


        phone = check_phone(
            data.contact
        )


        check_location(
            data.location
        )


        if not data.crops:

            raise ValueError(
                "Add at least one crop"
            )


        # Convert the field size before saving.

        field_size = (
            standardize_field_size(

                data.fieldSize.value,

                data.fieldSize.unit

            )
        )


        # Convert every crop before saving anything.

        prepared_crops = []


        for crop in data.crops:

            crop_name = (
                crop.crop
                .strip()
            )


            if not crop_name:

                raise ValueError(
                    "Crop name is required"
                )


            quantities = (
                standardize_crop_quantity(

                    crop.quantity,

                    crop.unit

                )
            )


            prepared_crops.append(
                {

                    "name":
                        crop_name.title(),

                    "entered_quantity":
                        crop.quantity,

                    "entered_unit":
                        crop.unit,

                    "kg":
                        quantities["kg"],

                    "quintal":
                        quantities[
                            "quintal"
                        ],

                    "tonne":
                        quantities[
                            "tonne"
                        ]

                }
            )


        connection = get_connection()

        cursor = connection.cursor()


        try:

            # Save the farmer.

            cursor.execute(
                """
                INSERT INTO farmers (

                    name,

                    contact,

                    location_text,

                    latitude,

                    longitude

                )

                VALUES (?, ?, ?, ?, ?)
                """,

                (

                    data.name.strip(),

                    phone,

                    data.location.text.strip(),

                    data.location.lat,

                    data.location.lng

                )
            )


            farmer_id = (
                cursor.lastrowid
            )


            # Save field information.

            cursor.execute(
                """
                INSERT INTO fields (

                    farmer_id,

                    location_text,

                    latitude,

                    longitude,

                    entered_size,

                    entered_unit,

                    size_acres,

                    size_hectares

                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,

                (

                    farmer_id,

                    data.location.text.strip(),

                    data.location.lat,

                    data.location.lng,

                    data.fieldSize.value,

                    data.fieldSize.unit,

                    field_size[
                        "acres"
                    ],

                    field_size[
                        "hectares"
                    ]

                )
            )


            field_id = (
                cursor.lastrowid
            )


            crop_ids = []


            # Save every crop row from the frontend.

            for crop in prepared_crops:

                cursor.execute(
                    """
                    INSERT INTO farmer_crops (

                        farmer_id,

                        field_id,

                        crop_name,

                        entered_quantity,

                        entered_unit,

                        quantity_kg,

                        quantity_quintal,

                        quantity_tonne

                    )

                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,

                    (

                        farmer_id,

                        field_id,

                        crop[
                            "name"
                        ],

                        crop[
                            "entered_quantity"
                        ],

                        crop[
                            "entered_unit"
                        ],

                        crop[
                            "kg"
                        ],

                        crop[
                            "quintal"
                        ],

                        crop[
                            "tonne"
                        ]

                    )
                )


                crop_ids.append(
                    cursor.lastrowid
                )


            connection.commit()


        except Exception:

            connection.rollback()

            raise


        finally:

            connection.close()


        return {

            "id":
                farmer_id,

            "farmerId":
                farmer_id,

            "fieldId":
                field_id,

            "cropIds":
                crop_ids,

            "message":
                "Farmer registered successfully"

        }


    except ValueError as error:

        raise HTTPException(

            status_code=400,

            detail=str(error)

        )


# ============================================================
# TRANSPORTER REGISTRATION
# ============================================================
#
# This endpoint matches your HTML:
#
# POST /api/transporters/register
# ============================================================


@app.post(
    "/api/transporters/register"
)
def register_transporter(
    data: TransporterRegistration
):

    try:

        if not data.name.strip():

            raise ValueError(
                "Enter your name or firm name"
            )


        phone = check_phone(
            data.contact
        )


        check_location(
            data.location
        )


        currency = (
            data.currency
            .strip()
            .upper()
        )


        if currency != "INR":

            raise ValueError(
                "Only INR is supported"
            )


        connection = get_connection()

        cursor = connection.cursor()


        cursor.execute(
            """
            INSERT INTO transporters (

                name,

                contact,

                location_text,

                latitude,

                longitude,

                min_price_per_tonne,

                currency

            )

            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,

            (

                data.name.strip(),

                phone,

                data.location.text.strip(),

                data.location.lat,

                data.location.lng,

                data.minPricePerTonne,

                currency

            )
        )


        connection.commit()


        transporter_id = (
            cursor.lastrowid
        )


        connection.close()


        return {

            "id":
                transporter_id,

            "transporterId":
                transporter_id,

            "message":
                "Transporter registered successfully"

        }


    except ValueError as error:

        raise HTTPException(

            status_code=400,

            detail=str(error)

        )


# ============================================================
# GET FARMER DETAILS
# ============================================================


@app.get(
    "/api/farmers/{farmer_id}"
)
def get_farmer(
    farmer_id: int
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *

        FROM farmers

        WHERE farmer_id = ?
        """,

        (
            farmer_id,
        )
    )


    farmer = cursor.fetchone()


    if farmer is None:

        connection.close()

        raise HTTPException(

            status_code=404,

            detail="Farmer not found"

        )


    cursor.execute(
        """
        SELECT *

        FROM fields

        WHERE farmer_id = ?
        """,

        (
            farmer_id,
        )
    )


    fields = cursor.fetchall()


    cursor.execute(
        """
        SELECT *

        FROM farmer_crops

        WHERE farmer_id = ?

        ORDER BY farmer_crop_id
        """,

        (
            farmer_id,
        )
    )


    crops = cursor.fetchall()

    connection.close()


    return {

        "farmer":
            dict(farmer),

        "fields": [

            dict(field)

            for field in fields

        ],

        "crops": [

            dict(crop)

            for crop in crops

        ]

    }


# ============================================================
# GET TRANSPORTERS
# ============================================================


@app.get(
    "/api/transporters"
)
def get_transporters():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *

        FROM transporters

        WHERE active = 1

        ORDER BY
            min_price_per_tonne ASC
        """
    )


    rows = cursor.fetchall()

    connection.close()


    return [

        dict(row)

        for row in rows

    ]


# ============================================================
# ADD A MARKET
# ============================================================


@app.post(
    "/api/markets"
)
def add_market(
    data: MarketInput
):

    if not -90 <= data.lat <= 90:

        raise HTTPException(
            status_code=400,
            detail="Latitude is not valid"
        )


    if not -180 <= data.lng <= 180:

        raise HTTPException(
            status_code=400,
            detail="Longitude is not valid"
        )


    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        INSERT INTO markets (

            market_name,

            city,

            district,

            state,

            latitude,

            longitude

        )

        VALUES (?, ?, ?, ?, ?, ?)
        """,

        (

            data.marketName.strip(),

            data.city.strip(),

            data.district.strip(),

            data.state.strip(),

            data.lat,

            data.lng

        )
    )


    connection.commit()

    market_id = cursor.lastrowid

    connection.close()


    return {

        "id":
            market_id,

        "message":
            "Market added successfully"

    }


# ============================================================
# GET MARKETS
# ============================================================


@app.get(
    "/api/markets"
)
def get_markets():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *

        FROM markets

        WHERE active = 1

        ORDER BY
            state,
            district,
            market_name
        """
    )


    markets = cursor.fetchall()

    connection.close()


    return [

        dict(market)

        for market in markets

    ]


# ============================================================
# DISTANCE
# ============================================================


def calculate_distance(
    lat1,
    lng1,
    lat2,
    lng2
):

    # Radius of Earth in kilometres.

    earth_radius = (
        6371.0088
    )


    lat1 = math.radians(
        lat1
    )


    lng1 = math.radians(
        lng1
    )


    lat2 = math.radians(
        lat2
    )


    lng2 = math.radians(
        lng2
    )


    difference_lat = (
        lat2 - lat1
    )


    difference_lng = (
        lng2 - lng1
    )


    value = (

        math.sin(
            difference_lat / 2
        ) ** 2

        +

        math.cos(lat1)

        * math.cos(lat2)

        * math.sin(
            difference_lng / 2
        ) ** 2

    )


    angle = (

        2

        * math.atan2(

            math.sqrt(value),

            math.sqrt(
                1 - value
            )

        )

    )


    return round(

        earth_radius
        * angle,

        2

    )


# ============================================================
# FIND NEARBY MARKETS
# ============================================================


def find_nearby_markets(
    latitude,
    longitude,
    maximum_distance
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *

        FROM markets

        WHERE active = 1
        """
    )


    markets = cursor.fetchall()

    connection.close()


    results = []


    for market in markets:

        distance = calculate_distance(

            latitude,

            longitude,

            market[
                "latitude"
            ],

            market[
                "longitude"
            ]

        )


        if distance <= maximum_distance:

            item = dict(
                market
            )


            item[
                "distance_km"
            ] = distance


            results.append(
                item
            )


    results.sort(

        key=lambda market:
            market[
                "distance_km"
            ]

    )


    return results


# ============================================================
# OLD PRICE DATA
# ============================================================


@app.post(
    "/api/prices"
)
def add_price(
    data: PriceInput
):

    try:

        date_value = (
            datetime.strptime(

                data.date,

                "%Y-%m-%d"

            )
        )


    except ValueError:

        raise HTTPException(

            status_code=400,

            detail="Date must be YYYY-MM-DD"

        )


    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT market_id

        FROM markets

        WHERE market_id = ?
        """,

        (
            data.marketId,
        )
    )


    if cursor.fetchone() is None:

        connection.close()

        raise HTTPException(

            status_code=404,

            detail="Market not found"

        )


    cursor.execute(
        """
        INSERT INTO historical_prices (

            market_id,

            crop_name,

            price_date,

            month,

            year,

            minimum_price,

            maximum_price,

            modal_price,

            previous_price,

            supply_quantity,

            demand_index

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,

        (

            data.marketId,

            data.crop
            .strip()
            .title(),

            data.date,

            date_value.month,

            date_value.year,

            data.minimumPrice,

            data.maximumPrice,

            data.modalPrice,

            data.previousPrice,

            data.supplyQuantity,

            data.demandIndex

        )
    )


    connection.commit()

    price_id = cursor.lastrowid

    connection.close()


    # Remove the old saved model because we now
    # have new training information.

    if MODEL_FILE.exists():

        MODEL_FILE.unlink()


    return {

        "id":
            price_id,

        "message":
            "Price record added"

    }


def load_price_data():

    connection = get_connection()


    data = pd.read_sql_query(
        """
        SELECT *

        FROM historical_prices

        ORDER BY price_date
        """,

        connection
    )


    connection.close()


    return data


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================


def train_price_model():

    data = load_price_data()


    if len(data) < 10:

        raise ValueError(
            "At least 10 old price records are needed"
        )


    data = data.dropna(
        subset=[

            "market_id",

            "crop_name",

            "month",

            "year",

            "modal_price"

        ]
    )


    data[
        "previous_price"
    ] = (

        data[
            "previous_price"
        ]

        .fillna(
            data[
                "modal_price"
            ]
        )

    )


    data[
        "supply_quantity"
    ] = (

        data[
            "supply_quantity"
        ]
        .fillna(0)

    )


    data[
        "demand_index"
    ] = (

        data[
            "demand_index"
        ]
        .fillna(1)

    )


    # Things used to make the prediction.

    X = data[
        [

            "market_id",

            "crop_name",

            "month",

            "year",

            "previous_price",

            "supply_quantity",

            "demand_index"

        ]
    ]


    # Price the model learns from.

    y = data[
        "modal_price"
    ]


    # Crop name is text so it has to be converted
    # before Random Forest can use it.

    processor = ColumnTransformer(
        transformers=[

            (

                "crop",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                [
                    "crop_name"
                ]

            ),

            (

                "numbers",

                "passthrough",

                [

                    "market_id",

                    "month",

                    "year",

                    "previous_price",

                    "supply_quantity",

                    "demand_index"

                ]

            )

        ]
    )


    model = Pipeline(
        steps=[

            (

                "prepare",

                processor

            ),

            (

                "forest",

                RandomForestRegressor(

                    n_estimators=250,

                    max_depth=15,

                    min_samples_split=2,

                    random_state=42

                )

            )

        ]
    )


    model.fit(
        X,
        y
    )


    # Save the model so it doesn't have to train
    # every time the farmer checks a market.

    joblib.dump(
        model,
        MODEL_FILE
    )


    return model


# ============================================================
# TRAIN MODEL ROUTE
# ============================================================


@app.post(
    "/api/model/train"
)
def train_model_route():

    try:

        model = (
            train_price_model()
        )


        return {

            "message":
                "Price model trained successfully",

            "recordsUsed":
                len(
                    load_price_data()
                )

        }


    except ValueError as error:

        raise HTTPException(

            status_code=400,

            detail=str(error)

        )


# ============================================================
# LOAD MODEL
# ============================================================


def load_price_model():

    if MODEL_FILE.exists():

        return joblib.load(
            MODEL_FILE
        )


    return train_price_model()


# ============================================================
# GET LATEST KNOWN PRICE
# ============================================================


def get_latest_price(
    market_id,
    crop
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *

        FROM historical_prices

        WHERE
            market_id = ?

        AND
            LOWER(crop_name)
            =
            LOWER(?)

        ORDER BY
            price_date DESC

        LIMIT 1
        """,

        (

            market_id,

            crop

        )
    )


    row = cursor.fetchone()

    connection.close()


    if row is None:

        return None


    return dict(
        row
    )


# ============================================================
# PREDICT CROP PRICE
# ============================================================


def predict_price(
    market_id,
    crop,
    month
):

    check_month(
        month
    )


    latest = get_latest_price(

        market_id,

        crop

    )


    if latest is None:

        return None


    model = (
        load_price_model()
    )


    prediction_data = pd.DataFrame(
        [
            {

                "market_id":
                    market_id,

                "crop_name":
                    crop
                    .strip()
                    .title(),

                "month":
                    month,

                "year":
                    datetime.now().year,

                "previous_price":
                    latest[
                        "modal_price"
                    ],

                "supply_quantity":
                    latest[
                        "supply_quantity"
                    ]
                    or 0,

                "demand_index":
                    latest[
                        "demand_index"
                    ]
                    or 1

            }
        ]
    )


    prediction = (
        model.predict(
            prediction_data
        )[0]
    )


    return round(
        float(prediction),
        2
    )


# ============================================================
# FIND TRANSPORTERS
# ============================================================


def get_active_transporters():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *

        FROM transporters

        WHERE active = 1
        """
    )


    rows = cursor.fetchall()

    connection.close()


    return [

        dict(row)

        for row in rows

    ]


def calculate_transport_cost(
    transporter,
    quantity_tonnes
):

    # The transporter frontend asks them for
    # their minimum price per tonne.

    cost = (

        quantity_tonnes

        * transporter[
            "min_price_per_tonne"
        ]

    )


    return round(
        cost,
        2
    )


def find_best_transporter(
    quantity_tonnes,
    farmer_latitude=None,
    farmer_longitude=None
):

    transporters = (
        get_active_transporters()
    )


    if not transporters:

        return None


    options = []


    for transporter in transporters:

        item = dict(
            transporter
        )


        item[
            "estimated_cost"
        ] = calculate_transport_cost(

            transporter,

            quantity_tonnes

        )


        # If both sides pinned their locations,
        # we can also show the distance.

        if (

            farmer_latitude is not None

            and farmer_longitude is not None

            and transporter[
                "latitude"
            ] is not None

            and transporter[
                "longitude"
            ] is not None

        ):

            item[
                "distance_from_farmer_km"
            ] = calculate_distance(

                farmer_latitude,

                farmer_longitude,

                transporter[
                    "latitude"
                ],

                transporter[
                    "longitude"
                ]

            )


        else:

            item[
                "distance_from_farmer_km"
            ] = None


        options.append(
            item
        )


    # Lowest transport price first.

    options.sort(

        key=lambda item:
            item[
                "estimated_cost"
            ]

    )


    return options[0]


# ============================================================
# ADD STORAGE
# ============================================================


@app.post(
    "/api/storage"
)
def add_storage(
    data: StorageInput
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT market_id

        FROM markets

        WHERE market_id = ?
        """,

        (
            data.marketId,
        )
    )


    if cursor.fetchone() is None:

        connection.close()

        raise HTTPException(

            status_code=404,

            detail="Market not found"

        )


    cursor.execute(
        """
        INSERT INTO storage_facilities (

            market_id,

            storage_name,

            available_capacity_quintal,

            cost_per_quintal_per_day,

            contact

        )

        VALUES (?, ?, ?, ?, ?)
        """,

        (

            data.marketId,

            data.name,

            data.availableCapacityQuintal,

            data.costPerQuintalPerDay,

            data.contact

        )
    )


    connection.commit()

    storage_id = cursor.lastrowid

    connection.close()


    return {

        "id":
            storage_id,

        "message":
            "Storage added"

    }


def find_storage(
    market_id,
    quantity_quintal
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *

        FROM storage_facilities

        WHERE
            market_id = ?

        AND
            active = 1

        AND
            available_capacity_quintal
            >= ?

        ORDER BY
            cost_per_quintal_per_day ASC

        LIMIT 1
        """,

        (

            market_id,

            quantity_quintal

        )
    )


    storage = cursor.fetchone()

    connection.close()


    if storage is None:

        return None


    return dict(
        storage
    )


# ============================================================
# MARKET RECOMMENDATION
# ============================================================


@app.post(
    "/api/markets/recommend"
)
def recommend_market(
    request: RecommendationInput
):

    try:

        check_month(
            request.sellingMonth
        )


        if request.storageDays < 0:

            raise ValueError(
                "Storage days cannot be negative"
            )


        quantities = (
            standardize_crop_quantity(

                request.quantity,

                request.unit

            )
        )


        connection = get_connection()

        cursor = connection.cursor()


        cursor.execute(
            """
            SELECT *

            FROM farmers

            WHERE farmer_id = ?
            """,

            (
                request.farmerId,
            )
        )


        farmer = cursor.fetchone()

        connection.close()


        if farmer is None:

            raise ValueError(
                "Farmer not found"
            )


        farmer = dict(
            farmer
        )


        # Nearby-market calculation needs GPS.

        if (

            farmer[
                "latitude"
            ] is None

            or farmer[
                "longitude"
            ] is None

        ):

            raise ValueError(
                "Pin the farm location before checking nearby markets"
            )


        markets = (
            find_nearby_markets(

                farmer[
                    "latitude"
                ],

                farmer[
                    "longitude"
                ],

                request.maximumDistanceKm

            )
        )


        if not markets:

            raise ValueError(
                "No nearby markets were found"
            )


        transporter = (
            find_best_transporter(

                quantities[
                    "tonne"
                ],

                farmer[
                    "latitude"
                ],

                farmer[
                    "longitude"
                ]

            )
        )


        if transporter is None:

            raise ValueError(
                "No transporter is registered yet"
            )


        results = []


        # Check every nearby market.

        for market in markets:

            predicted_price = (
                predict_price(

                    market[
                        "market_id"
                    ],

                    request.crop,

                    request.sellingMonth

                )
            )


            # Skip markets where this crop has
            # no old price data.

            if predicted_price is None:

                continue


            gross_value = (

                quantities[
                    "quintal"
                ]

                * predicted_price

            )


            transport_cost = (
                transporter[
                    "estimated_cost"
                ]
            )


            storage = None

            storage_cost = 0


            if request.storageDays > 0:

                storage = find_storage(

                    market[
                        "market_id"
                    ],

                    quantities[
                        "quintal"
                    ]

                )


                if storage is None:

                    continue


                storage_cost = (

                    quantities[
                        "quintal"
                    ]

                    * request.storageDays

                    * storage[
                        "cost_per_quintal_per_day"
                    ]

                )


            total_cost = (

                transport_cost

                + storage_cost

            )


            expected_amount = (

                gross_value

                - total_cost

            )


            results.append(
                {

                    "marketId":
                        market[
                            "market_id"
                        ],

                    "marketName":
                        market[
                            "market_name"
                        ],

                    "city":
                        market[
                            "city"
                        ],

                    "district":
                        market[
                            "district"
                        ],

                    "state":
                        market[
                            "state"
                        ],

                    "distanceKm":
                        market[
                            "distance_km"
                        ],

                    "predictedPricePerQuintal":
                        round(
                            predicted_price,
                            2
                        ),

                    "grossValue":
                        round(
                            gross_value,
                            2
                        ),

                    "transportCost":
                        round(
                            transport_cost,
                            2
                        ),

                    "storageCost":
                        round(
                            storage_cost,
                            2
                        ),

                    "totalCost":
                        round(
                            total_cost,
                            2
                        ),

                    "expectedAmount":
                        round(
                            expected_amount,
                            2
                        ),

                    "transporter": {

                        "id":
                            transporter[
                                "transporter_id"
                            ],

                        "name":
                            transporter[
                                "name"
                            ],

                        "contact":
                            transporter[
                                "contact"
                            ],

                        "minimumPricePerTonne":
                            transporter[
                                "min_price_per_tonne"
                            ],

                        "currency":
                            transporter[
                                "currency"
                            ],

                        "distanceFromFarmerKm":
                            transporter[
                                "distance_from_farmer_km"
                            ]

                    },

                    "storage":
                        dict(storage)
                        if storage
                        else None

                }
            )


        if not results:

            raise ValueError(
                "There is not enough market data for this crop yet"
            )


        # Farmer keeps the most money in the first result.

        results.sort(

            key=lambda item:
                item[
                    "expectedAmount"
                ],

            reverse=True

        )


        for position, item in enumerate(
            results,
            start=1
        ):

            item[
                "rank"
            ] = position


        best = (
            results[0]
        )


        # Save the best recommendation.

        connection = get_connection()

        cursor = connection.cursor()


        cursor.execute(
            """
            INSERT INTO recommendations (

                farmer_id,

                crop_name,

                market_id,

                transporter_id,

                predicted_price,

                gross_value,

                transport_cost,

                storage_cost,

                total_cost,

                expected_amount,

                distance_km

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (

                request.farmerId,

                request.crop
                .strip()
                .title(),

                best[
                    "marketId"
                ],

                best[
                    "transporter"
                ][
                    "id"
                ],

                best[
                    "predictedPricePerQuintal"
                ],

                best[
                    "grossValue"
                ],

                best[
                    "transportCost"
                ],

                best[
                    "storageCost"
                ],

                best[
                    "totalCost"
                ],

                best[
                    "expectedAmount"
                ],

                best[
                    "distanceKm"
                ]

            )
        )


        connection.commit()

        recommendation_id = (
            cursor.lastrowid
        )

        connection.close()


        return {

            "recommendationId":
                recommendation_id,

            "crop":
                request.crop
                .strip()
                .title(),

            "standardQuantity":
                quantities,

            "recommendedMarket":
                best,

            "marketComparison":
                results

        }


    except ValueError as error:

        raise HTTPException(

            status_code=400,

            detail=str(error)

        )


# ============================================================
# CREATE TRANSACTION
# ============================================================


@app.post(
    "/api/transactions"
)
def create_transaction(
    request: TransactionInput
):

    try:

        quantities = (
            standardize_crop_quantity(

                request.quantity,

                request.unit

            )
        )


        connection = get_connection()

        cursor = connection.cursor()


        # Check farmer.

        cursor.execute(
            """
            SELECT *

            FROM farmers

            WHERE farmer_id = ?
            """,

            (
                request.farmerId,
            )
        )


        farmer = cursor.fetchone()


        if farmer is None:

            connection.close()

            raise ValueError(
                "Farmer not found"
            )


        # Check transporter.

        cursor.execute(
            """
            SELECT *

            FROM transporters

            WHERE
                transporter_id = ?

            AND
                active = 1
            """,

            (
                request.transporterId,
            )
        )


        transporter = (
            cursor.fetchone()
        )


        if transporter is None:

            connection.close()

            raise ValueError(
                "Transporter not found"
            )


        transporter = dict(
            transporter
        )


        transport_price = (
            calculate_transport_cost(

                transporter,

                quantities[
                    "tonne"
                ]

            )
        )


        cursor.execute(
            """
            INSERT INTO transactions (

                farmer_id,

                transporter_id,

                market_id,

                crop_name,

                quantity_kg,

                transport_price,

                status

            )

            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,

            (

                request.farmerId,

                request.transporterId,

                request.marketId,

                request.crop
                .strip()
                .title(),

                quantities[
                    "kg"
                ],

                transport_price,

                "CREATED"

            )
        )


        transaction_id = (
            cursor.lastrowid
        )


        connection.commit()

        connection.close()


        return {

            "id":
                transaction_id,

            "transportPrice":
                transport_price,

            "currency":
                transporter[
                    "currency"
                ],

            "status":
                "CREATED"

        }


    except ValueError as error:

        raise HTTPException(

            status_code=400,

            detail=str(error)

        )


# ============================================================
# SHIPMENT STATUSES
# ============================================================


SHIPMENT_STATUSES = [

    "TRANSPORT BOOKED",

    "VEHICLE ASSIGNED",

    "VEHICLE REACHED FARM",

    "CROP COLLECTED",

    "IN TRANSIT",

    "REACHED MARKET",

    "UNLOADING",

    "DELIVERED",

    "SALE COMPLETED"

]


# ============================================================
# CREATE SHIPMENT
# ============================================================


@app.post(
    "/api/transactions/{transaction_id}/shipment"
)
def create_shipment(
    transaction_id: int
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *

        FROM transactions

        WHERE transaction_id = ?
        """,

        (
            transaction_id,
        )
    )


    transaction = cursor.fetchone()


    if transaction is None:

        connection.close()

        raise HTTPException(

            status_code=404,

            detail="Transaction not found"

        )


    cursor.execute(
        """
        INSERT INTO shipments (

            transaction_id,

            current_status

        )

        VALUES (?, ?)
        """,

        (

            transaction_id,

            "TRANSPORT BOOKED"

        )
    )


    shipment_id = (
        cursor.lastrowid
    )


    # Also save the first tracking update.

    cursor.execute(
        """
        INSERT INTO shipment_updates (

            shipment_id,

            status,

            message

        )

        VALUES (?, ?, ?)
        """,

        (

            shipment_id,

            "TRANSPORT BOOKED",

            "Transport has been booked"

        )
    )


    # Change transaction status too.

    cursor.execute(
        """
        UPDATE transactions

        SET status = ?

        WHERE transaction_id = ?
        """,

        (

            "TRANSPORT BOOKED",

            transaction_id

        )
    )


    connection.commit()

    connection.close()


    return {

        "id":
            shipment_id,

        "status":
            "TRANSPORT BOOKED"

    }


# ============================================================
# UPDATE SHIPMENT
# ============================================================


@app.put(
    "/api/shipments/{shipment_id}/status"
)
def update_shipment(
    shipment_id: int,
    request: ShipmentUpdateInput
):

    status = (
        request.status
        .strip()
        .upper()
    )


    if status not in SHIPMENT_STATUSES:

        raise HTTPException(

            status_code=400,

            detail="Shipment status is not valid"

        )


    if (

        request.lat is not None

        and not -90 <= request.lat <= 90

    ):

        raise HTTPException(

            status_code=400,

            detail="Latitude is not valid"

        )


    if (

        request.lng is not None

        and not -180 <= request.lng <= 180

    ):

        raise HTTPException(

            status_code=400,

            detail="Longitude is not valid"

        )


    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT transaction_id

        FROM shipments

        WHERE shipment_id = ?
        """,

        (
            shipment_id,
        )
    )


    shipment = cursor.fetchone()


    if shipment is None:

        connection.close()

        raise HTTPException(

            status_code=404,

            detail="Shipment not found"

        )


    transaction_id = (
        shipment[
            "transaction_id"
        ]
    )


    now = (
        datetime.now()
        .isoformat()
    )


    # Change current shipment status.

    cursor.execute(
        """
        UPDATE shipments

        SET
            current_status = ?,

            last_updated = ?

        WHERE shipment_id = ?
        """,

        (

            status,

            now,

            shipment_id

        )
    )


    # Keep the old updates too.

    cursor.execute(
        """
        INSERT INTO shipment_updates (

            shipment_id,

            status,

            message,

            latitude,

            longitude

        )

        VALUES (?, ?, ?, ?, ?)
        """,

        (

            shipment_id,

            status,

            request.message,

            request.lat,

            request.lng

        )
    )


    # Keep transaction and shipment in sync.

    cursor.execute(
        """
        UPDATE transactions

        SET status = ?

        WHERE transaction_id = ?
        """,

        (

            status,

            transaction_id

        )
    )


    # Finish the transaction once sale is complete.

    if status == "SALE COMPLETED":

        cursor.execute(
            """
            UPDATE transactions

            SET
                status = 'COMPLETED',

                completed_at = ?

            WHERE transaction_id = ?
            """,

            (

                now,

                transaction_id

            )
        )


    connection.commit()

    connection.close()


    return {

        "id":
            shipment_id,

        "status":
            status

    }


# ============================================================
# TRACK SHIPMENT
# ============================================================


@app.get(
    "/api/shipments/{shipment_id}"
)
def track_shipment(
    shipment_id: int
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT

            s.*,

            tr.farmer_id,

            tr.transporter_id,

            tr.market_id,

            tr.crop_name,

            tr.quantity_kg,

            tr.transport_price,

            t.name AS transporter_name,

            t.contact AS transporter_contact,

            m.market_name,

            m.city

        FROM shipments s

        JOIN transactions tr

        ON
            s.transaction_id
            =
            tr.transaction_id

        JOIN transporters t

        ON
            tr.transporter_id
            =
            t.transporter_id

        LEFT JOIN markets m

        ON
            tr.market_id
            =
            m.market_id

        WHERE
            s.shipment_id = ?
        """,

        (
            shipment_id,
        )
    )


    shipment = cursor.fetchone()


    if shipment is None:

        connection.close()

        raise HTTPException(

            status_code=404,

            detail="Shipment not found"

        )


    cursor.execute(
        """
        SELECT

            update_id,

            status,

            message,

            latitude,

            longitude,

            update_time

        FROM shipment_updates

        WHERE shipment_id = ?

        ORDER BY
            update_time ASC,

            update_id ASC
        """,

        (
            shipment_id,
        )
    )


    updates = cursor.fetchall()

    connection.close()


    return {

        "shipment":
            dict(shipment),

        "trackingHistory": [

            dict(update)

            for update in updates

        ]

    }


# ============================================================
# FARMER TRANSACTION HISTORY
# ============================================================


@app.get(
    "/api/farmers/{farmer_id}/transactions"
)
def farmer_transaction_history(
    farmer_id: int
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT

            tr.transaction_id,

            tr.crop_name,

            tr.quantity_kg,

            tr.transport_price,

            tr.status,

            tr.created_at,

            tr.completed_at,

            t.name AS transporter_name,

            t.contact AS transporter_contact,

            m.market_name,

            m.city,

            s.shipment_id,

            s.current_status

        FROM transactions tr

        JOIN transporters t

        ON
            tr.transporter_id
            =
            t.transporter_id

        LEFT JOIN markets m

        ON
            tr.market_id
            =
            m.market_id

        LEFT JOIN shipments s

        ON
            tr.transaction_id
            =
            s.transaction_id

        WHERE
            tr.farmer_id = ?

        ORDER BY
            tr.created_at DESC
        """,

        (
            farmer_id,
        )
    )


    rows = cursor.fetchall()

    connection.close()


    return [

        dict(row)

        for row in rows

    ]


# ============================================================
# BACKEND STATUS
# ============================================================


@app.get(
    "/api"
)
def api_status():

    return {

        "name":
            "Fasal-Net",

        "backend":
            "running",

        "message":
            "Fasal-Net backend is working"

    }


# ============================================================
# SERVE THE HTML FRONTEND
# ============================================================


@app.get(
    "/"
)
def frontend():

    if not FRONTEND_FILE.exists():

        raise HTTPException(

            status_code=404,

            detail=(
                "static/index.html was not found"
            )

        )


    return FileResponse(
        str(FRONTEND_FILE)
    )
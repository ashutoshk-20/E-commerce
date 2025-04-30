import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

// Import Images from Assets
import Headphone1 from "../../assets/Head1.png";
import Headphone2 from "../../assets/Head2.png";
import Headphone3 from "../../assets/Head3.png";
import Headphone4 from "../../assets/Head4.png";
import Headphone5 from "../../assets/Head5.png";

import Laptop1 from "../../assets/Lap1.png";
import Laptop2 from "../../assets/Lap2.png";
import Laptop3 from "../../assets/Lap3.png";
import Laptop4 from "../../assets/Lap4.png";
import Laptop5 from "../../assets/Lap5.png";

import iPhone1 from "../../assets/I1.png";
import iPhone2 from "../../assets/I2.png";
import iPhone3 from "../../assets/I3.png";
import iPhone4 from "../../assets/I4.png";
import iPhone5 from "../../assets/I5.png";

import Android1 from "../../assets/A1.png";
import Android2 from "../../assets/A2.png";
import Android3 from "../../assets/A3.png";
import Android4 from "../../assets/A4.png";
import Android5 from "../../assets/A5.png";

const categoryData = {
    Headphones: [
        { id: 1, slug: "boat-rockerz-480", name: "Boat Rockerz 480", img: Headphone1, price: "₹1799.00" },
        { id: 2, slug:"jbl-t-450",name: "JBL TODE 450BT", img: Headphone2, price: "₹5500.00" },
        { id: 3, slug: "boat-rockerz-350", name: "Boat Rockerz 350", img: Headphone3, price: "₹1199.00" },
        { id: 4, slug:"jbl-tune-750", name: "JBL TUNE 750BT", img: Headphone4, price: "₹6500.00" },
        { id: 5, slug: "boat-rockerz-450", name: "Boat Rockerz 450 Pro", img: Headphone5, price: "₹1799.00" },
    ],
    Laptops: [
        { id: 1, slug: "lenovo-yoga", name: "LENOVO YOGA", img: Laptop1, price: "₹1,77,999.00" },
        { id: 2, slug: "dell-xps", name: "Dell XPS 13", img: Laptop2, price: "₹70,999.00" },
        { id: 3, slug: "hp-spectre", name: "HP Spectre x360", img: Laptop3, price: "₹87,900.00" },
        { id: 4, slug: "macbook-pro", name: "Apple MacBook Pro", img: Laptop4, price: "₹1,00,000.00" },
        { id: 5, slug: "macbook-air", name: "Apple MacBook Air M3", img: Laptop5, price: "₹1,50,000.00" },
    ],
    iPhones: [
        { id: 1, slug: "iphone-16", name: "iPhone 16", img: iPhone1, price: "₹80,000.00" },
        { id: 2, slug: "iphone-16-pro", name: "iPhone 16 Pro", img: iPhone2, price: "₹1,19,000.00" },
        { id: 3, slug: "iphone-15-pro", name: "iPhone 15 Pro", img: iPhone3, price: "₹1,10,000.00" },
        { id: 4, slug: "iphone-14-pro", name: "iPhone 14 Pro", img: iPhone4, price: "₹99,000.00" },
        { id: 5, slug: "iphone-16e", name: "iPhone 16e", img: iPhone5, price: "₹59,990.00" },
    ],
    Androids: [
        { id: 1, slug: "samsung-s25-ultra", name: "Samsung S25 Ultra", img: Android1, price: "₹1,65,999.00" },
        { id: 2, slug: "samsung-s24-ultra", name: "Samsung S24 Ultra", img: Android2, price: "₹1,45,990.00" },
        { id: 3, slug: "samsung-s25", name: "Samsung S25", img: Android3, price: "₹80,990.00" },
        { id: 4, slug: "google-pixel-9-pro", name: "Google Pixel 9 Pro", img: Android4, price: "₹1,20,000.00" },
        { id: 5, slug:"google-pixel-8a", name: "Google Pixel 8a 5G", img: Android5, price: "₹85,000.00" },
    ],
};

const Categories = () => {
    const [selectedCategory, setSelectedCategory] = useState("Headphones");
    const navigate = useNavigate();

    return (
        <div id="categories" className="container flex flex-col items-center mt-12 space-y-6">
            <h2 className="text-primary font-semibold tracking-widest text-2xl uppercase sm:text-3xl">
                Explore Our Categories
            </h2>

            {/* Category Selection Buttons */}
            <div className="flex gap-4">
                {Object.keys(categoryData).map((category) => (
                    <button
                        key={category}
                        onClick={() => setSelectedCategory(category)}
                        className={`px-4 py-2 rounded-lg font-semibold transition-all duration-300 ${selectedCategory === category
                            ? "bg-primary text-white"
                            : "bg-gray-200 hover:bg-primary hover:text-white"
                            }`}
                    >
                        {category}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-12">
                {categoryData[selectedCategory].map((product) => (
                    <div
                        key={product.slug}
                        onClick={() => navigate(`/product/${selectedCategory}/${product.slug}`)}
                        className="w-full max-w-[320px] p-6 px-12 border rounded-2xl shadow-xl flex flex-col items-center text-center transition-transform duration-300 will-change-transform hover:scale-105 cursor-pointer"
                    >
                        <img src={product.img} alt={product.name} className="w-40 h-40 object-contain mb-4" />
                        <h3 className="text-xl font-bold">{product.name}</h3>
                        <span className="text-primary font-bold mt-3 text-2xl">{product.price}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Categories;
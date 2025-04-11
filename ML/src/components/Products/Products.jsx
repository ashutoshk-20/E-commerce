import React from "react";
import Product1 from "../../assets/Product1.png";
import Product2 from "../../assets/Product2.png";
import Product3 from "../../assets/Product3.png";
import Image4 from "../../assets/Image4.png";
import Product5 from "../../assets/Product5.png";
import Product6 from "../../assets/Product6.png";

const products = [
    { id: 1, name: "HEADPHONES", img: Product1, description: "Wireless Bluetooth headphones with deep bass and long battery life." },
    { id: 2, name: "TWS", img: Product2, description: "Compact, noise-isolating earbuds with immersive sound." },
    { id: 3, name: "MACBOOKS", img: Product3, description: "Sleek and powerful laptop with a Retina display and M-series chip." },
    { id: 4, name: "IPHONES", img: Image4, description: "Advanced smartphone with a dynamic display and top-tier camera." },
    { id: 5, name: "ANDROIDS", img: Product5, description: "Flagship phone with an epic camera and super-fast performance." },
    { id: 6, name: "VR's", img: Product6, description: "Cutting-edge mixed reality headset for an immersive experience." },
];

const Products = () => {
    return (
        <div id="products" className="container flex flex-col items-center mt-12 space-y-6">
            <h2 className="text-primary font-semibold tracking-widest text-2xl uppercase sm:text-3xl">
                Our Products
            </h2>

            {/* Product Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
                {products.map((product) => (
                    <div
                        key={product.id}
                        className="p-6 border rounded-lg shadow-lg flex flex-col items-center text-center transition-transform duration-300 hover:scale-105 hover:shadow-2xl cursor-pointer"
                    >
                        <img
                            src={product.img}
                            alt={product.name}
                            className="w-40 h-40 object-contain mb-4 transition-transform duration-300 hover:scale-110"
                        />
                        <h3 className="text-xl font-semibold">{product.name}</h3>
                        <p className="text-gray-600">{product.description}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Products;
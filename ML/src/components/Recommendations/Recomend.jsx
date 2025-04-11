import React, { useEffect, useState } from "react";
import { useContext } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import ProductContext from "../context/ProductContext";

const Recommend = ({ category }) => {
    const { products } = useContext(ProductContext);
    const [related, setRelated] = useState([]);
    const navigate = useNavigate();

    useEffect(() => {
        if (products.length > 0) {
            let filteredProducts = products.filter((item) => item.category === category);
            setRelated(filteredProducts.slice(0, 5));
        }
    }, [products, category]);

    // Function to handle click with animation & scroll to top
    const handleProductClick = (item) => {
        document.body.style.transition = "opacity 0.3s ease-in-out";
        document.body.style.opacity = "0";

        setTimeout(() => {
            navigate(`/product/${item.category}/${item.slug}`);
            document.body.style.opacity = "1";
            window.scrollTo({ top: 0, behavior: "smooth" }); // Scroll to top smoothly
        }, 300);
    };

    return (
        <div className="my-24">
            <h2 className="text-red-600 font-semibold tracking-widest text-2xl uppercase sm:text-3xl text-center">
                OTHER RECOMMENDATIONS
            </h2><br />

            <div className='grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4'>
                {related.map((item, index) => (
                    <motion.div 
                        key={index}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="border p-3 rounded-lg hover:shadow-lg transition cursor-pointer"
                        onClick={() => handleProductClick(item)}
                    >
                        <img src={item.image[0]} alt={item.name} className="w-full h-40 object-contain" />
                        <h3 className="text-lg font-semibold mt-2">{item.name}</h3>
                        <p className="text-red-600 font-bold">{item.price}</p>
                    </motion.div>
                ))}
            </div>
        </div>
    );
};

export default Recommend;
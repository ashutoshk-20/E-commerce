import { useParams } from "react-router-dom";
import { useContext, useEffect, useState } from "react";
import ProductContext from "../context/ProductContext";
import Recommend from "../Recommendations/Recomend";

const ProductDetails = () => {
    const { slug } = useParams();
    const { products } = useContext(ProductContext);
    const [productData, setProductData] = useState(null);
    const [image, setImage] = useState('');

    useEffect(() => {
        if (products.length > 0) {
            const foundProduct = products.find(item => item.slug === slug);
            if (foundProduct) {
                setProductData(foundProduct);
                setImage(foundProduct.image[0]);  // Set default image
            } else {
                setProductData(null);
            }
        }
    }, [slug, products]);

    if (!productData) {
        return <div className="text-center text-gray-600 text-lg mt-20">Loading product details...</div>;
    }

    return (
        <div className="border-t-2 pt-10 transition-opacity ease-in-out duration-500 opacity-100">
            {/* Product Data  */}
            <div className="flex flex-col sm:flex-row gap-12">

                {/* Product Image */}
                <div className="flex-1 flex flex-col-reverse sm:flex-row gap-1">
                    {/* Sidebar Thumbnails */}
                    <div className="flex sm:flex-col overflow-x-auto sm:overflow-y-scroll sm:w-[16%] w-full">
                        {productData.image.map((img, index) => (
                            <img
                                onClick={() => setImage(img)}
                                src={img}
                                key={index}
                                className="w-[22%] sm:w-full cursor-pointer rounded-md border p-0.5 hover:scale-105 transition-transform"
                                alt={productData.name}
                            />
                        ))}
                    </div>
                    {/* Main Image */}
                    <div className="w-full sm:w-[84%]">
                        <img className="w-full h-auto rounded-lg border p-1" src={image} alt={productData.name} />
                    </div>
                </div>
                {/* Product Info */}
                <div className="flex-1">
                    <h1 className="font-semibold text-3xl mt-2">{productData.name}</h1>
                    <div className="flex items-center gap-1 mt-2 text-yellow-500 text-lg">
                        {"★★★★★"}
                        <p className="pl-2 text-black">(122)</p>
                    </div>
                    <p className="mt-5 text-3xl font-semibold text-red-600">{productData.price}</p>
                    <ul className="mt-5 text-lg text-gray-700 md:w-4/5 list-disc pl-5 space-y-2">
                        {productData.description.map((point, index) => (
                            <li key={index}>{point}</li>
                        ))}
                    </ul>

                    {/* Spacing between description and button */}
                    <button className="mt-6 px-8 py-3 bg-red-600 text-white text-lg font-semibold rounded-md transition duration-300 hover:scale-105 hover:bg-red-700">
                        ADD TO CART
                    </button>

                    <hr className="mt-8 sm:w-4/5" />
                    <div className="text-md text-gray-600 mt-5 flex flex-col gap-1">
                        <p>✅ 100% Original Product</p>
                        <p>✅ Cash On Delivery Available</p>
                        <p>✅ Fast Return & Exchange within 21 Days!</p>
                    </div>
                </div>
            </div>

            {/* Related Products Section */}
            <Recommend category={productData.category} />
        </div>
    );
};

export default ProductDetails;
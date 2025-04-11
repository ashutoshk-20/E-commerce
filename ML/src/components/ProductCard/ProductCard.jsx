const ProductCard = ({ product }) => {
    return (
      <div className="max-w-sm bg-white shadow-md rounded-xl overflow-hidden border border-gray-200 p-4 transition hover:shadow-lg">
        <img src={product.image[0]} alt={product.name} className="w-full h-48 object-contain" />
        <div className="mt-4">
          <h3 className="text-lg font-bold text-gray-900">{product.name}</h3>
          <p className="text-primary font-semibold">{product.price}</p>
          <p className="text-sm text-gray-500 capitalize">{product.category}</p>
        </div>
      </div>
    );
  };
  
  export default ProductCard;
  
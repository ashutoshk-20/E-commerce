import React from "react";

const SearchBar = ({ searchTerm, setSearchTerm }) => {
  return (
    <div className="flex flex-col items-center mt-8 space-y-3">
      <h2 className='text-primary font-semibold tracking-widest text-2xl uppercase sm:text-3xl'>
        Search Your Product From Thousands of Products
      </h2>

      <div className="w-full max-w-2xl flex bg-white rounded-full shadow-lg overflow-hidden border-2 border-primary">
        <input 
          type="text" 
          placeholder="Search for products..." 
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full px-6 py-3 text-lg outline-none text-gray-700"
        />
        <button 
          className="bg-primary text-white px-6 py-3 font-semibold hover:bg-opacity-80 transition-all"
        >
          Search
        </button>
      </div>
    </div>
  );
};

export default SearchBar;

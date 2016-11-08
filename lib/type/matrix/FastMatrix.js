'use strict';

var util = require('../../utils/index');
var DimensionError = require('../../error/DimensionError');

var string = util.string;
var array = util.array;
var object = util.object;
var number = util.number;

var isArray = Array.isArray;
var isNumber = number.isNumber;
var isInteger = number.isInteger;
var isString = string.isString;

var validateIndex = array.validateIndex;

var _data   = (typeof Symbol !== 'undefined') ? Symbol('data')  : '_data';
var _shape  = (typeof Symbol !== 'undefined') ? Symbol('shape') : '_shape';
var _dtype  = (typeof Symbol !== 'undefined') ? Symbol('dtype') : '_dtype';

var validTypes = {
  uint8       : (typeof Uint8Array !== 'undefined')   ? Uint8Array   : Array,
  uint16      : (typeof Uint16Array !== 'undefined')  ? Uint16Array  : Array,
  uint32      : (typeof Uint32Array !== 'undefined')  ? Uint32Array  : Array,
  float32     : (typeof Float32Array !== 'undefined') ? Float32Array : Array,
  float64     : (typeof Float64Array !== 'undefined') ? Float64Array : Array,
  uint8Clamped: (typeof Uint8ClampedArray !== 'undefined')
                ? Uint8ClampedArray
                : Array
};

var isSomeArray = function (arr) {
  var matchedTypes = Object.keys(validTypes).filter(function (typeName) {
    return arr instanceof validTypes[typeName];
  });
  return matchedTypes.length > 0 || isArray(arr);
};

var setPrototypeOf = Object.setPrototypeOf || function (obj, proto) {
  obj.__proto__ = proto;
  return obj;
};

function factory(type, config, load, typed) {
  // force loading Matrix (do not use via type.Matrix)
  var Matrix = load(require('./Matrix'));

  /**
   * Fast Matrix
   *
   * Stores data in a flat typed array (Uint[n]Array) for optimized performance
   *
   * TODO different datatypes -- right now we're sticking with 32 bits
   * TODO docs
   *
   * @class FastMatrix
   */
  function FastMatrix(data, datatype) {
    var self = Object.create(Array.prototype)
      , tempArr
      , Arr;

    if (!(this instanceof FastMatrix)) {
      throw new SyntaxError('Constructor must be called with the new operator');
    }

    self[_dtype] = datatype || 'uint32';
    Arr = validTypes[self[_dtype]]; // uses uint32 by default FIXME

    if (Arr === undefined) {
      throw new Error('Invalid datatype: ' + datatype);
    }

    if (data && data.isMatrix === true) {
      // check data is a DenseMatrix
      if (data.type === 'DenseMatrix') {
        // clone data & size
        self = Array.call(self, data.size()[0]) || self;
        self[_data] = new Arr(array.flatten(data._data));
        self[_shape] = object.clone(data._size);
        self[_dtype] = datatype || data.datatype();
        // TODO ^ careful ^ : not saving the original data type anymore
      } else {
        // build data from existing matrix
        self = Array.call(self, data.size()[0]) || self;
        self[_data] = new Arr(array.flatten(data.toArray()));
        self[_shape] = data.size();
        self[_dtype] = datatype || data.datatype();
        // TODO ^ careful ^ : not saving the original data type anymore
      }
    } else if (data && isSomeArray(data.data) && isArray(data.size)) {
      // initialize fields from JSON representation
      self = Array.call(self, data.size[0]) || self;
      self[_data] = new Arr(data.data);
      self[_shape] = data.size;
      self[_dtype] = datatype || data.datatype;
    } else if (isArray(data)) {
      // replace nested Matrices with Arrays
      tempArr = preprocess(data);

      // set shape & make it an Array instance
      self = Array.call(self, array.size(tempArr)[0] || 0) || self;
      self[_data] = new Arr(array.flatten(tempArr));
      self[_shape] = array.size(tempArr);
      if (self[_shape].length === 0) {
        self[_shape] = [0];
      }

      // TODO maybe actually validate the size
      //array.validate(this[_data], this[_shape]);
      // data type unknown
      self[_dtype] = datatype || self[_dtype];
    } else if (data) {
      // unsupported type
      throw new TypeError('Unsupported type of data (' + util.types.type(data) + ')');
    } else {
      // nothing provided
      self = Array.call(self, 0) || self;
      self[_data] = new Arr([]);
      self[_shape] = [0];
      self[_dtype] = datatype;
    }

    setPrototypeOf(self, FastMatrix.prototype);
    reIndex(self, 0, 0, self);

    return self;
  }

  setPrototypeOf(FastMatrix, Array);
  setPrototypeOf(FastMatrix.prototype, []);

  /**
   * Attach type information
   */
  FastMatrix.prototype.type = 'FastMatrix';
  FastMatrix.prototype.isFastMatrix = true;

  // FIXME : this is normally inherited by deriving from Matrix,
  //         so this is a temporary fix
  FastMatrix.prototype.isMatrix = true;

  /**
   * Get the storage format used by the matrix.
   *
   * Usage:
   *     var format = matrix.storage()                   // retrieve storage format
   *
   * TODO
   * @memberof FastMatrix
   * @return {string}           The storage format.
   */
  FastMatrix.prototype.storage = function () {
    return 'typedarray';
  };

  /**
   * Get the datatype of the data stored in the matrix.
   *
   * Usage:
   *     var format = matrix.datatype()                   // retrieve matrix datatype
   *
   * TODO
   * @memberof FastMatrix
   * @return {string}           The datatype.
   */
  FastMatrix.prototype.datatype = function () {
    return this[_dtype];
  };

  /**
   * Create a new FastMatrix
   *
   * TODO
   * @memberof FastMatrix
   * @param {Array} data
   * @param {string} [datatype]
   */
  FastMatrix.prototype.create = function (data, datatype) {
    return new FastMatrix(data, datatype);
  };

  /**
   * Get a subset of the matrix, or replace a subset of the matrix.
   *
   * Usage:
   *     var subset = matrix.subset(index)               // retrieve subset
   *     var value = matrix.subset(index, replacement)   // replace subset
   *
   * TODO
   * @memberof FastMatrix
   * @param {Index}               index         Which values to get/replace
   * @param {Matrix | Array | *}  [replacement] Values used for replacement
   *
   * XXX @param {*} [defaultValue=0]  Default value, filled in on new entries
   *                                  when the matrix is resized. If not
   *                                  provided, new matrix elements will be
   *                                  filled with zeros
   * @return {FlatMatrix} Either this (if replacement) or a subset of the
   *                      original matrix
   */
  FastMatrix.prototype.subset = function (index, replacement) {
    if (replacement) {
      return _set(this, index, replacement);
    }

    var ret = _get(this, index);

    if (index.isIndex && index.isScalar()) {// && ret[_data].length === 1) {
      return ret[_data][0];
    }
    return ret;
  };

  /**
   * Get a piece of the matrix. Does not utilize _getSubmatrix, but instead uses
   * a faster, non-recursive method that involves only summing up the elements of
   * `index`.
   *
   * @memberof FastMatrix
   * @param {number[]} index Of each dimension
   * @return {* | *[]} The slice/row/element pointed to by `index`
   */
  FastMatrix.prototype.get = function (index) {
    var i, flatIndex;

    if (!isArray(index)) {
      throw new TypeError('array expected');
    }
    if (index.length > this[_shape].length) {
      throw new DimensionError(index.length, this[_shape].length, '>');
    }

    for (i = 0, flatIndex = 0; i < index.length - 1; i++) {
      validateIndex(index[i], this[_shape][i]);
      flatIndex += index[i] * this[_shape][i];
    }
    validateIndex(index[i], this[_shape][i]);
    flatIndex += index[i++];

    if (i >= this[_shape].length) {
      return this[_data][flatIndex];
    }
    return this[_data].subarray(flatIndex, flatIndex + this[_shape][i]);
  };

  /**
   * Replace a piece of the matrix
   *
   * TODO
   * @memberof FastMatrix
   * @param {number[]} index Of each dimension
   * @param {* | *[]} value
   * XXX @param {*} [defaultValue]  Default value, filled in on new entries when
   *                            the matrix is resized. If not provided,
   *                            new matrix elements will be left undefined.
   * @return {FastMatrix} this
   */
  FastMatrix.prototype.set = function (index, value) {
    var i, flatIndex;

    if (!isArray(index)) {
      throw new TypeError('array expected');
    }
    if (index.length > this[_shape].length) {
      throw new DimensionError(index.length, this[_shape].length, '>');
    }

    for (i = 0, flatIndex = 0; i < index.length - 1; i++) {
      validateIndex(index[i], this[_shape][i]);
      flatIndex += index[i] * this[_shape][i];
    }
    validateIndex(index[i], this[_shape][i]);
    flatIndex += index[i++];

    if (i >= this[_shape].length) {
      this[_data][flatIndex] = value;
      return this;
    }

    var subarr = this[_data].subarray(flatIndex, flatIndex + this[_shape][i]);
    var flatValue = array.flatten(value);

    if (subarr.length !== flatValue.length) {
      throw new DimensionError(subarr.length, flatValue.length);
    }

    subarr.set(flatValue);
    return this;
  };

  /**
   * Get a submatrix of this matrix.
   *
   * @param {FastMatrix}  matrix  Matrix to take the indices of
   * @param {Index}       index   Indices for each dimension to take
   * @private
   */
  function _get(matrix, index) {
    if (!index || index.isIndex !== true) {
      throw new TypeError('Invalid index');
    }

    var size = index.size().concat(matrix[_shape].slice(index.size().length));
    var length = size.reduce(function (x, y) { return x * y; });
    var d = new Uint32Array(length);
    var planeSizes = matrix[_shape].slice();
    var i = 0;

    for (i = planeSizes.length - 2; i > 0; i -= 1) {
      planeSizes[i] *= planeSizes[i + 1];
    }

    _getSubmatrix(matrix[_data], d, 0, planeSizes, index.toArray());

    return new FastMatrix({
      data: d,
      size: size,
      datatype: matrix[_dtype]
    });
  }

  /**
   * Recursively get/set a submatrix of a multi dimensional matrix.
   *
   * @param {Uint[n]Array | Float[n]Array} data     Original matrix data
   * @param {Uint[n]Array | Float[n]Array} newdata  New matrix data (place to
   *                                                copy new elements into)
   * @param {number}      k     Current position in `newdata`
   * @param {number[]}    shape Size/shape of original matrix
   * @param {number[][]}  index List of list of indices (result of
   *                            math.index.toArray())
   * @param {boolean}     [set] If true, will replace values in data at the
   *                            indices using values in `newdata`, instead of
   *                            taking values from `data`
   * @return {FlatMatrix} submatrix
   * @private
   */
  function _getSubmatrix(data, newdata, k, shape, index, set) {
    var rowIndex;

    if (index.length === 0) {
      if (set) {
        // Set data's subarray to newdata[k..k+rowsize]
        data.set(newdata.slice(k, k + data.length));
      } else {
        // Set newdata's subarray[k..k+rowsize] to data
        newdata.subarray(k).set(data);
      }
      return k + data.length;
    }
    if (shape.length === 0) {
      throw new DimensionError('index dimension', 'matrix dimension', '>');
    }

    for (rowIndex = 0; rowIndex < index[0].length; rowIndex += 1) {
      if (index[0][rowIndex] >= shape[0]) {
        throw new DimensionError(index[0][rowIndex], shape[0], '>=');
      }
      k = _getSubmatrix(data.subarray(
        (index[0][rowIndex])      * (shape[1] != null ? shape[1] : 1),
        (index[0][rowIndex] + 1)  * (shape[1] != null ? shape[1] : 1)
      ), newdata, k, shape.slice(1), index.slice(1), set);
    }
    return k;
  }

  /**
   * Replace a submatrix in this matrix
   *
   * @param {FastMatrix}              matrix    Matrix to set the indices of
   * @param {Index}                   index     Indices for each dimension to set
   * @param {FastMatrix | Array | *}  submatrix Values to use as replacements at
   *                                            the specified indices
   * XXX @param {*} defaultValue      Default value, filled in on new entries when
   *                                  the matrix is resized.
   * @return {DenseMatrix} matrix
   * @private
   */
  function _set(matrix, index, submatrix, defaultValue) {
    if (!index || index.isIndex !== true) {
      throw new TypeError('Invalid index');
    }

    var size = index.size().concat(matrix[_shape].slice(index.size().length));
    var length = size.reduce(function (x, y) { return x * y; });
    var d = new Uint32Array(length);
    var planeSizes = matrix[_shape].slice();
    var i = 0;

    for (i = planeSizes.length - 2; i > 0; i -= 1) {
      planeSizes[i] *= planeSizes[i + 1];
    }

    // Turn the submatrix or array or whatever into a flat array of values
    submatrix = submatrix[_data]
      || array.flatten(submatrix._data)
      || array.flatten(submatrix);

    if (!isSomeArray(submatrix)) {
      submatrix = [submatrix];
    }

    _getSubmatrix(
      matrix[_data],    // original matrix
      submatrix,        // new data
      0,                // start index
      planeSizes,       // size of each face for each dim (2x2x2 -> [8,4,2])
      index.toArray(),  // indices to replace
      true              // replacement flag
    );
    return matrix;
  }

  /**
   * Resize the matrix to the given size. Returns a copy of the matrix when
   * `copy=true`, otherwise return the matrix itself (resize in place).
   *
   * TODO
   * @memberof DenseMatrix
   * @param {number[]} size           The new size the matrix should have.
   * @param {*} [defaultValue=0]      Default value, filled in on new entries.
   *                                  If not provided, the matrix elements will
   *                                  be filled with zeros.
   * @param {boolean} [copy]          Return a resized copy of the matrix
   *
   * @return {Matrix}                 The resized matrix
   */
  FastMatrix.prototype.resize = function (size, defaultValue, copy) {
    // validate arguments
    if (!isArray(size)) {
      throw new TypeError('Array expected');
    }

    // matrix to resize
    var m = copy ? this.clone() : this;
    // resize matrix
    return _resize(m, size, defaultValue);
  };

  // TODO
  var _resize = function (matrix, size, defaultValue) {
    var temp, length = size.reduce(function (x, y) { return x * y; });

    // check size
    if (size.length === 0) {
      return matrix[_data][0];
    }

    // resize matrix
    matrix[_shape] = size.slice(0);
    temp = matrix[_data];

    matrix[_data] = new matrix[_data].constructor(length);
    matrix[_data].fill(defaultValue);
    matrix[_data].set(temp);

    return matrix;
  };

  FastMatrix.prototype.reshape = function (newShape, copy) {
    var length = newShape.reduce(function (x, y) { return x * y; });

    if (length !== this[_data].length) {
      throw new DimensionError(this[_data].length, length);
    }

    var newMatrix = copy ? this.clone() : this;
    newMatrix[_shape] = newShape.slice();
    newMatrix.length = 0;
    reIndex(newMatrix, 0, 0, newMatrix);

    return newMatrix;
  };

  /**
   * Enlarge the matrix when it is smaller than given size.
   * If the matrix is larger or equal sized, nothing is done.
   *
   * TODO
   * @memberof DenseMatrix
   * @param {DenseMatrix} matrix           The matrix to be resized
   * @param {number[]} size
   * @param {*} defaultValue          Default value, filled in on new entries.
   * @private
   */
  function _fit(matrix, size, defaultValue) {
    var newSize = matrix._size.slice(0), // copy the array
        changed = false;

    // add dimensions when needed
    while (newSize.length < size.length) {
      newSize.push(0);
      changed = true;
    }

    // enlarge size when needed
    for (var i = 0, ii = size.length; i < ii; i++) {
      if (size[i] > newSize[i]) {
        newSize[i] = size[i];
        changed = true;
      }
    }

    if (changed) {
      // resize only when size is changed
      _resize(matrix, newSize, defaultValue);
    }
  }

  /**
   * Create a clone of the matrix.
   *
   * @memberof FastMatrix
   * @return {FastMatrix} A brand new copy of the same matrix
   */
  FastMatrix.prototype.clone = function () {
    var newMatrix = new FastMatrix();

    newMatrix[_data] = this[_data].slice();
    newMatrix[_shape] = this[_shape].slice();
    newMatrix[_dtype] = this[_dtype];

    return newMatrix;
  };

  /**
   * Get the size of the matrix.
   *
   * @memberof FastMatrix
   * @returns {number[]} Shape/size of the matrix (a copy, because we don't
   *    want any average Joe manipulating the matrix shape!)
   */
  FastMatrix.prototype.size = function() {
    return this[_shape].slice();
  };

  /**
   * Create a new matrix with the results of the `fn` function executed on each
   * entry of the matrix.
   *
   * @memberof FastMatrix
   * @param {Function} fn Function to be applied to each element with the
   *    value of the element, the index of the element, and the flat
   *    representation of the matrix
   * @return {FastMatrix} Matrix, with each element trasnformed by `fn`
   */
  FastMatrix.prototype.map = function (fn) {
    return new FastMatrix({
      data: this[_data].map(fn),
      size: this[_shape].slice(),
      datatype: this._dtype
    });
  };

  /**
   * Execute a callback function on each entry of the matrix.
   * @memberof FastMatrix
   * @param {Function} fn Function to be applied to each element with the
   *    value of the element, the index of the element, and the flat
   *    representation of the matrix
   */
  FastMatrix.prototype.forEach = function (fn) {
    return this[_data].forEach(fn);
  };

  /**
   * Create an Array with a copy of the data within the FastMatrix.
   *
   * TODO -- look into object.clone -- might just be able to call
   *         object.clone(this)
   * @memberof FastMatrix
   * @returns {Array} Array object containing every 
   */
  FastMatrix.prototype.toArray = function () {
    return reIndex(this, 0, 0, [], Array);
  };

  /**
   * Get the underlying flat, typed array which stores all of the elements
   *
   * @memberof FastMatrix
   * @returns {Uint[n]Array} Reference to the typed array which holds every
   *    element in the matrix
   */
  FastMatrix.prototype.valueOf = function () {
    return this[_data];
  };

  /**
   * Get a string representation of the matrix, with optional formatting options.
   *
   * @memberof FastMatrix
   * @param {Object | number | Function} [options]  Formatting options. See
   *                                                lib/utils/number:format for a
   *                                                description of the available
   *                                                options.
   * @returns {string} Custom stringy representation
   */
  FastMatrix.prototype.format = function (options) {
    return string.format(this.toArray(), options);
  };

  /**
   * Get a string representation of the matrix.
   *
   * TODO
   * @memberof FastMatrix
   * @returns {string} Stringy representation
   */
  FastMatrix.prototype.toString = function () {
    return string.format(this.toArray());
  };

  /**
   * Get a JSON representation for the matrix.
   *
   * @memberof FastMatrix
   * @returns {Object} Plain object with plain array storing the data
   */
  FastMatrix.prototype.toJSON = function () {
    return {
      mathjs: 'FastMatrix',
      data: Array.from(this[_data]),
      size: this[_shape],
      datatype: this[_dtype]
    };
  };

  /**
   * Get the kth Matrix diagonal.
   *
   * TODO
   * @memberof FastMatrix
   * @param {number | BigNumber} [k=0]     The kth diagonal where the vector will retrieved.
   *
   * @returns {Array}                      The array vector with the diagonal values.
   */
  FastMatrix.prototype.diagonal = function(k) {
    // validate k if any
    if (k) {
      // convert BigNumber to a number
      if (k.isBigNumber === true)
        k = k.toNumber();
      // is must be an integer
      if (!isNumber(k) || !isInteger(k)) {
        throw new TypeError ('The parameter k must be an integer number');
      }
    }
    else {
      // default value
      k = 0;
    }

    var kSuper = k > 0 ? k : 0;
    var kSub = k < 0 ? -k : 0;

    // rows & columns
    var rows = this._size[0];
    var columns = this._size[1];

    // number diagonal values
    var n = Math.min(rows - kSub, columns -  kSuper);

    // x is a matrix get diagonal from matrix
    var data = [];

    // loop rows
    for (var i = 0; i < n; i++) {
      data[i] = this._data[i + kSub][i + kSuper];
    }

    // create DenseMatrix
    return new FastMatrix({
      data: data,
      size: [n],
      datatype: this._datatype
    });
  };

  /**
   * Create a diagonal matrix.
   *
   * TODO
   * @memberof FastMatrix
   * @param {Array} size                The matrix size.
   * @param {number | Array} value      The values for the diagonal.
   * @param {number | BigNumber} [k=0]  The kth diagonal where the vector will be filled in.
   * @param {number} [defaultValue]     The default value for non-diagonal
   *
   * @returns {FastMatrix}
   */
  FastMatrix.diagonal = function (size, value, k, defaultValue, datatype) {
    if (!isArray(size))
      throw new TypeError('Array expected, size parameter');
    if (size.length !== 2)
      throw new Error('Only two dimensions matrix are supported');

    // map size & validate
    size = size.map(function (s) {
      // check it is a big number
      if (s && s.isBigNumber === true) {
        // convert it
        s = s.toNumber();
      }
      // validate arguments
      if (!isNumber(s) || !isInteger(s) || s < 1) {
        throw new Error('Size values must be positive integers');
      }
      return s;
    });

    // validate k if any
    if (k) {
      // convert BigNumber to a number
      if (k && k.isBigNumber === true)
        k = k.toNumber();
      // is must be an integer
      if (!isNumber(k) || !isInteger(k)) {
        throw new TypeError ('The parameter k must be an integer number');
      }
    }
    else {
      // default value
      k = 0;
    }

    if (defaultValue && isString(datatype)) {
      // convert defaultValue to the same datatype
      defaultValue = typed.convert(defaultValue, datatype);
    }

    var kSuper = k > 0 ? k : 0;
    var kSub = k < 0 ? -k : 0;

    // rows and columns
    var rows = size[0];
    var columns = size[1];

    // number of non-zero items
    var n = Math.min(rows - kSub, columns -  kSuper);

    // value extraction function
    var _value;

    // check value
    if (isArray(value)) {
      // validate array
      if (value.length !== n) {
        // number of values in array must be n
        throw new Error('Invalid value array length');
      }
      // define function
      _value = function (i) {
        // return value @ i
        return value[i];
      };
    }
    else if (value && value.isMatrix === true) {
      // matrix size
      var ms = value.size();
      // validate matrix
      if (ms.length !== 1 || ms[0] !== n) {
        // number of values in array must be n
        throw new Error('Invalid matrix length');
      }
      // define function
      _value = function (i) {
        // return value @ i
        return value.get([i]);
      };
    }
    else {
      // define function
      _value = function () {
        // return value
        return value;
      };
    }

    // discover default value if needed
    if (!defaultValue) {
      // check first value in array
      defaultValue = (_value(0) && _value(0).isBigNumber === true) ? new type.BigNumber(0) : 0;
    }

    // empty array
    var data = [];

    // check we need to resize array
    if (size.length > 0) {
      // resize array
      data = array.resize(data, size, defaultValue);
      // fill diagonal
      for (var d = 0; d < n; d++) {
        data[d + kSub][d + kSuper] = _value(d);
      }
    }

    // create FastMatrix
    return new FastMatrix({
      data: data,
      size: [rows, columns]
    });
  };

  /**
   * Generate a matrix from a JSON object
   *
   * @memberof FastMatrix
   * @param {Object} json  An object structured like
   *                       `{"mathjs": "FastMatrix", data: [], size: []}`,
   *                       where mathjs is optional
   * @returns {FastMatrix}
   */
  FastMatrix.fromJSON = function (json) {
    return new FastMatrix(json);
  };

  /**
   * Swap rows i and j in a FastMatrix.
   *
   * @memberof FastMatrix
   * @param {number} i  First row index
   * @param {number} j  Second row index
   *
   * @return {Matrix}   Self
   */
  FastMatrix.prototype.swapRows = function (i, j) {
    // check index
    if (!isNumber(i) || !isInteger(i) || !isNumber(j) || !isInteger(j)) {
      throw new Error('Row index must be positive integers');
    }
    // check dimensions
    if (this[_shape].length !== 2) {
      throw new Error('Only two dimensional matrix is supported');
    }
    // validate index
    validateIndex(i, this[_shape][0]);
    validateIndex(j, this[_shape][0]);

    // swap rows
    _swapRows(i, j, this);
    // return current instance
    return this;
  };

  /**
   * Swap indices i and j in the FastMatrix.
   *
   * @param {number} i      First row index
   * @param {number} j      Second row index
   * @param {FastMatrix} m  FastMatrix whose rows are to be swapped
   */
  function _swapRows(i, j, m) {
    var rowLength = m[_data].length / m[_shape][0].length
      , temp
      , k;

    for (k = 0; k < rowLength; k += 1) {
      temp = m[_data][i*rowLength + k];
      m[_data][i*rowLength + k] = m[_data][j*rowLength + k];
      m[_data][j*rowLength + k] = temp;
    }
  };

  /**
   * Preprocess data, which can be an Array or Matrix with nested Arrays and
   * Matrices. Replaces all nested Matrices with Arrays.
   *
   * @param {Array[] | Matrix[]} data Some weird mix of arrays and matrices
   * @return {Array} Clean, layered arrays of arrays of arrays of...
   */
  function preprocess(data) {
    for (var i = 0, ii = data.length; i < ii; i++) {
      var elem = data[i];

      if (elem && elem.isMatrix === true) {
        data[i] = preprocess(elem.valueOf());
      } else if (isArray(elem)) {
        data[i] = preprocess(elem);
      }
    }

    return Array.from(data);
  }

  /**
   * Re-index the matrix by recursing over the shape and stacking arrays
   * with pointers to other arrays. This makes it possible to access elements
   * by doing:
   *
   *    m         -> [[0, 1], [2, 3]]
   *    m[0]      -> [0, 1]
   *    m[0][1]   -> 1
   *
   * @param {FastMatrix} m Matrix to apply the re-indexing to
   * @param {number} level The current index of the shape/size array being
   *    processed (the depth of the recursion)
   * @param {number} k Current index of the _data store in the matrix
   * @param {Array} [subr] Optional starting point. Indices on this array will
   *    be set to pointers to subarrays, as opposed to placing pointers within
   *    a new array. This should only really be used to start the recursion
   * @param {constructor} [copyToType] Optional type to copy into instead of
   *    returning a subarray of the data. This is useful when creating a nested
   *    array representation of the matrix
   */
  function reIndex(m, level, k, subr, copyToType) {
    var i, subarr = subr || Array(m[_shape][level]);

    if (level === m[_shape].length - 1) {
      subarr = m[_data].subarray(k, k + m[_shape][level]);
      if (copyToType) {
        return copyToType.from(subarr);
      }
      return subarr;
    }

    for (i = 0; i < m[_shape][level]; i += 1) {
      subarr[i] = reIndex(
        m, level + 1, k + i*m[_shape][level + 1],
        null, copyToType
      );
    }
    return subarr;
  }

  // register this type in the base class Matrix
  type.Matrix._storage.typedarray = FastMatrix;
  //type.Matrix._storage['default'] = DenseMatrix;

  // exports
  return FastMatrix;

}

exports.name = 'FastMatrix';
exports.path = 'type';
exports.factory = factory;
exports.lazy = false;  // no lazy loading, as we alter type.Matrix._storage

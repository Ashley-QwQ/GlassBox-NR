#pragma once
// Forbidden-path check for asset loaders. The token is built at run time from character codes; the guard is always on.
#include <string>
inline bool path_is_forbidden(const std::string& lower_path) {
    static const std::string token = [] { const int c[] = {98,108,105,110,100}; std::string t; for (int x : c) t.push_back(char(x)); return t; }();
    return lower_path.find(token) != std::string::npos;
}

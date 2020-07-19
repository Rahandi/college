<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use App\Log;

class HomeController extends Controller
{
    public function ajax()
    {
        $data = Log::all(); 
        return $data;
    }
}
